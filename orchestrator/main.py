import asyncio
import os
import logging
import json
import aiohttp
from livekit import rtc, api
import numpy as np
from pydub import AudioSegment
import io

# --- Configuration ---
# Load configuration from environment variables for flexibility
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
ROOM_NAME = os.environ.get("ROOM_NAME", "bangla-voice-agent")
BOT_IDENTITY = "bangla-bot"

# Service URLs are resolved by docker-compose's DNS
VAD_URL = os.environ.get("VAD_URL", "http://vad:8000/detect_speech")
STT_URL = os.environ.get("STT_URL", "http://stt:8001/transcribe")
TTS_URL = os.environ.get("TTS_URL", "http://tts:8002/synthesize")
GOVT_API_URL = os.environ.get("GOVT_API_URL", "http://114.130.116.74/govtchat/chat/stream")

# --- VAD & Speech Buffering Parameters ---
VAD_SAMPLE_RATE = 16000  # VAD model's expected sample rate
VAD_CHUNK_DURATION_MS = 30  # VAD works well on 30ms chunks
VAD_CHUNK_SIZE = int(VAD_SAMPLE_RATE * (VAD_CHUNK_DURATION_MS / 1000.0))
SILENCE_DURATION_MS = 700  # How much silence indicates end of speech
MAX_SPEECH_DURATION_S = 20  # Max duration of a single speech segment

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("orchestrator")

class ParticipantProcessor:
    """
    Manages the entire conversation lifecycle for a single participant.
    This includes VAD, buffering, STT, API calls, TTS, and audio playback.
    """
    def __init__(self, room: rtc.Room, participant: rtc.RemoteParticipant, audio_source: rtc.AudioSource):
        self.room = room
        self.participant = participant
        self.audio_source = audio_source
        self.resampler = rtc.AudioResampler(rtc.AudioFormat.S16, rtc.AudioFormatType.LINEAR, 48000, 2, VAD_SAMPLE_RATE, 1)
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_duration_ms = 0
        self.processing_lock = asyncio.Lock()
        self.session = aiohttp.ClientSession()

    async def process_audio_stream(self, stream: rtc.AudioStream):
        """Main loop to process incoming audio frames from a participant."""
        async for frame in stream:
            if self.processing_lock.locked():
                continue  # Skip processing if a turn is already being handled

            resampled_frame = self.resampler.remix_and_resample(frame)
            await self.handle_vad(resampled_frame.data)

    async def handle_vad(self, frame_data: bytes):
        """Sends audio frame to VAD service and manages speech state."""
        try:
            async with self.session.post(VAD_URL, data=frame_data) as resp:
                if resp.status != 200:
                    logger.error(f"VAD service returned error: {resp.status}")
                    return
                vad_result = await resp.json()
                is_speech = vad_result.get("is_speech", False)

            if is_speech:
                self.silence_frames = 0
                if not self.is_speaking:
                    logger.info(f"Speech started for {self.participant.identity}")
                    self.is_speaking = True
                self.audio_buffer.extend(frame_data)
                self.speech_duration_ms += VAD_CHUNK_DURATION_MS
            elif self.is_speaking:
                self.silence_frames += 1
                silence_ms = self.silence_frames * VAD_CHUNK_DURATION_MS
                if silence_ms >= SILENCE_DURATION_MS:
                    logger.info(f"Speech ended for {self.participant.identity} after {silence_ms}ms of silence.")
                    asyncio.create_task(self.trigger_conversation_turn())

            if self.speech_duration_ms >= MAX_SPEECH_DURATION_S * 1000:
                logger.warning(f"Max speech duration reached for {self.participant.identity}")
                asyncio.create_task(self.trigger_conversation_turn())

        except aiohttp.ClientError as e:
            logger.error(f"Error communicating with VAD service: {e}")

    async def trigger_conversation_turn(self):
        """Initiates the full STT->API->TTS->Playback pipeline."""
        async with self.processing_lock:
            if not self.audio_buffer:
                logger.warning("Triggered conversation but buffer is empty.")
                self.reset_speech_state()
                return

            buffered_audio = self.audio_buffer
            self.reset_speech_state()

            try:
                # 1. Speech-to-Text
                transcribed_text = await self.transcribe_audio(buffered_audio)
                if not transcribed_text:
                    return

                # 2. Government API
                govt_response = await self.query_govt_api(transcribed_text)
                if not govt_response:
                    return

                # 3. Text-to-Speech
                synthesized_audio = await self.synthesize_speech(govt_response)
                if not synthesized_audio:
                    return

                # 4. Playback audio to room
                await self.play_audio_to_room(synthesized_audio)

            except Exception as e:
                logger.error(f"Error in conversation turn for {self.participant.identity}: {e}", exc_info=True)
            finally:
                logger.info(f"Conversation turn finished for {self.participant.identity}.")

    def reset_speech_state(self):
        """Resets the speaking state and buffer for the next turn."""
        self.is_speaking = False
        self.audio_buffer.clear()
        self.silence_frames = 0
        self.speech_duration_ms = 0

    async def transcribe_audio(self, audio_data: bytes) -> str | None:
        """Sends audio data to the STT service and returns the transcription."""
        logger.info(f"Sending {len(audio_data)} bytes to STT service...")
        form = aiohttp.FormData()
        form.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
        try:
            async with self.session.post(STT_URL, data=form) as resp:
                if resp.status != 200:
                    logger.error(f"STT service error: {resp.status} {await resp.text()}")
                    return None
                result = await resp.json()
                text = result.get("transcription", "").strip()
                logger.info(f"STT result: '{text}'")
                return text if text else None
        except aiohttp.ClientError as e:
            logger.error(f"Error communicating with STT service: {e}")
            return None

    async def query_govt_api(self, text: str) -> str | None:
        """Queries the external government API with the transcribed text."""
        logger.info(f"Querying government API with: '{text}'")
        payload = {"user_id": self.participant.identity, "query": text}
        full_response = ""
        try:
            async with self.session.post(GOVT_API_URL, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Govt API error: {resp.status} {await resp.text()}")
                    return None
                # Handle streaming response
                async for line in resp.content:
                    if line:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            if event.get("type") == "answer_chunk":
                                full_response += event.get("content", "")
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Govt API response: '{full_response[:100]}...'")
            return full_response
        except aiohttp.ClientError as e:
            logger.error(f"Error communicating with Govt API: {e}")
            return None

    async def synthesize_speech(self, text: str) -> bytes | None:
        """Sends text to the TTS service to get synthesized audio."""
        logger.info(f"Sending text to TTS service: '{text[:100]}...'")
        payload = {"text": text, "lang": "bn"}
        try:
            async with self.session.post(TTS_URL, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"TTS service error: {resp.status} {await resp.text()}")
                    return None
                audio_bytes = await resp.read()
                logger.info(f"Received {len(audio_bytes)} bytes of audio from TTS.")
                return audio_bytes
        except aiohttp.ClientError as e:
            logger.error(f"Error communicating with TTS service: {e}")
            return None

    async def play_audio_to_room(self, audio_bytes: bytes):
        """Decodes MP3 audio and plays it back into the LiveKit room."""
        logger.info("Playing synthesized audio response to the room.")
        try:
            # Decode MP3 bytes using pydub
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

            # Resample to the bot's output track format (48kHz) if needed
            segment = segment.set_frame_rate(48000).set_channels(1)

            # Get raw PCM data
            pcm_data = np.array(segment.get_array_of_samples())

            # Create audio frames and push them to the source
            frame_duration_ms = 20
            frame_size = int(48000 * (frame_duration_ms / 1000.0))

            for i in range(0, len(pcm_data), frame_size):
                chunk = pcm_data[i:i+frame_size]
                if len(chunk) < frame_size:
                    # Pad the last chunk if necessary
                    chunk = np.pad(chunk, (0, frame_size - len(chunk)), 'constant')

                frame = rtc.AudioFrame(
                    data=chunk.astype(np.int16).tobytes(),
                    sample_rate=48000,
                    num_channels=1,
                    samples_per_channel=frame_size
                )
                await self.audio_source.capture_frame(frame)
                await asyncio.sleep(frame_duration_ms / 1000.0 * 0.95) # Sleep to simulate real-time playback
        except Exception as e:
            logger.error(f"Error playing audio to room: {e}", exc_info=True)

    async def close(self):
        await self.session.close()

async def main():
    room = rtc.Room()
    processors = {}

    # The bot needs to publish its own audio track to send responses
    audio_source = rtc.AudioSource(48000, 1) # 48kHz, mono
    track = rtc.LocalAudioTrack.create_audio_track("bot-response-track", audio_source)

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    async def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        if participant.identity in processors:
            await processors[participant.identity].close()
            del processors[participant.identity]

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.AUDIO and participant.identity != BOT_IDENTITY:
            logger.info(f"Audio track subscribed from: {participant.identity}")
            if participant.identity not in processors:
                processors[participant.identity] = ParticipantProcessor(room, participant, audio_source)
            # Start a task to process this participant's audio stream
            asyncio.create_task(processors[participant.identity].process_audio_stream(track))

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(BOT_IDENTITY)
        .with_name("Bangla AI Agent")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
        .to_jwt()
    )

    try:
        logger.info(f"Connecting to LiveKit room '{ROOM_NAME}'...")
        await room.connect(LIVEKIT_URL, token)
        logger.info("Successfully connected to the room.")

        # Publish the bot's audio track after connecting
        await room.local_participant.publish_track(track)
        logger.info("Bot's audio track published.")

        # Keep the main function running to handle events
        await asyncio.Event().wait()

    except Exception as e:
        logger.error(f"Failed to run the bot: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting from the room.")
        await room.disconnect()
        for processor in processors.values():
            await processor.close()

if __name__ == "__main__":
    # Setup graceful shutdown
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        loop.close()
