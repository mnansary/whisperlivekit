import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gtts import gTTS, gTTSError
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic model for the request body to ensure type safety
class TTSRequest(BaseModel):
    text: str
    lang: str = 'bn'  # Default to Bangla ('bn')

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Accepts text and a language code, synthesizes speech using gTTS,
    and returns it as an MP3 audio stream.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        logger.info(f"Received synthesis request for lang='{request.lang}'. Text: '{request.text[:60]}...'")

        # Create an in-memory binary stream to hold the audio data
        audio_fp = io.BytesIO()

        # Initialize gTTS with the provided text and language
        tts = gTTS(text=request.text, lang=request.lang, slow=False)

        # Write the generated audio to the in-memory stream
        tts.write_to_fp(audio_fp)

        # Rewind the stream to the beginning so the client can read it from the start
        audio_fp.seek(0)

        logger.info("Speech synthesized successfully.")

        # Return the audio stream as a response.
        # The 'audio/mpeg' media type is correct for MP3 files.
        return StreamingResponse(audio_fp, media_type="audio/mpeg")

    except gTTSError as e:
        logger.error(f"gTTS Error: {e}. This might be due to an unsupported language or a network issue.")
        raise HTTPException(status_code=500, detail=f"Failed to synthesize speech: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during synthesis: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/health")
def health_check():
    """A simple health check endpoint to verify that the service is running."""
    return {"status": "ok", "service": "TTS"}
