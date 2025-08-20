# Real-Time Bangla Voice Agent

This project provides a complete, containerized system for a real-time, voice-driven AI agent that communicates in the Bangla language. The system is built on a microservices architecture, orchestrated with Docker Compose, and features a simple web-based frontend for user interaction.

## System Architecture

The system is composed of several independent services that work together to provide a seamless conversational experience.

```
[User's Browser] --(WebRTC Audio)--> [LiveKit Server] --(Audio Stream)--> [Orchestrator]
      ^                                                                         |
      |                                                                         | 1. VAD Check
      |                                                                         v
      +--------------------(Bot's Audio)---------------------------------- [VAD Service]
                                                                                |
                                                                                | 2. Speech Buffer
                                                                                v
                                                                        [Orchestrator]
                                                                                |
                                                                                | 3. Transcribe Request
                                                                                v
                                                                         [STT Service (GPU)]
                                                                                |
                                                                                | 4. Transcribed Text
                                                                                v
                                                                        [Orchestrator]
                                                                                |
                                                                                | 5. Query Govt API
                                                                                v
                                                                   [External Govt API Service]
                                                                                |
                                                                                | 6. API Response
                                                                                v
                                                                        [Orchestrator]
                                                                                |
                                                                                | 7. Synthesize Request
                                                                                v
                                                                         [TTS Service]
                                                                                |
                                                                                | 8. Synthesized Audio
                                                                                v
                                                                        [Orchestrator] -> (Publishes to LiveKit)

```

---

## Prerequisites

Before you begin, ensure you have the following installed on your host machine:
1.  **Docker Engine:** [Installation Guide](https://docs.docker.com/engine/install/)
2.  **Docker Compose:** [Installation Guide](https://docs.docker.com/compose/install/)
3.  **NVIDIA GPU Drivers:** Your host machine must have an NVIDIA GPU with the appropriate drivers installed.
4.  **NVIDIA Container Toolkit:** This is required to allow Docker containers to access the GPU. [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Setup Instructions

Follow these steps to configure and run the application:

**1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

**2. Create the Environment File**
The system uses a `.env` file to manage secret keys and configuration. An example file is provided.

```bash
# Copy the example file to create your own configuration
cp .env.example .env
```

**3. Generate a Frontend Token**
The frontend needs a long-lived token to connect to the LiveKit server. You must generate this token and add it to your `.env` file.

*   **Install the LiveKit CLI:** Follow the instructions at [docs.livekit.io/cli/](https://docs.livekit.io/cli/).
*   **Generate the token:** Run the following command in your terminal. This creates a token valid for 30 days (720 hours).
    ```bash
    livekit-cli create-token \
      --api-key devkey \
      --api-secret secret \
      --join \
      --room bangla-voice-agent \
      --identity user-frontend \
      --valid-for 720h
    ```
*   **Update `.env` file:** Copy the entire token string generated by the command and paste it as the value for `LIVEKIT_FRONTEND_TOKEN` in your `.env` file.

## How to Run

Once the setup is complete, you can build and run the entire system with a single command.

**1. Build and Start the Services**
```bash
# This command will build the images for our custom services and start all containers.
# The -d flag runs the services in detached mode (in the background).
docker-compose up --build -d
```
The first time you run this, it will take a while to download the base images and build the STT service, which includes the large Whisper model.

**2. Access the Frontend**
Open your web browser and navigate to:
[**http://localhost:8080**](http://localhost:8080)

You should see the "Bangla Voice Agent" interface with a "Connected" status.

**3. Monitor the Logs**
To see the real-time logs from all services, you can run:
```bash
docker-compose logs -f
```
To view logs for a specific service (e.g., the orchestrator), run:
```bash
docker-compose logs -f orchestrator
```

**4. Stop the Services**
To stop all running services, use the following command:
```bash
docker-compose down
```

---

## Service Explanations

*   **`livekit` (RTC Service):**
    *   **Technology:** `livekit/livekit-server`
    *   **Role:** The core WebRTC server that manages real-time audio and data connections between the frontend and the backend. It is the central hub for all media streams.

*   **`vad` (Voice Activity Detection Service):**
    *   **Technology:** Silero VAD, FastAPI
    *   **Role:** A lightweight service that listens to an audio stream and determines if it contains human speech. It exposes a REST endpoint (`/detect_speech`) that the Orchestrator calls to know when the user starts and stops speaking.

*   **`stt` (Speech-to-Text Service):**
    *   **Technology:** `insanely-fast-whisper` (using OpenAI's Whisper `large-v3` model), FastAPI
    *   **Role:** A high-performance, GPU-accelerated service for transcribing Bangla audio into text. It exposes a REST endpoint (`/transcribe`) that accepts an audio file and returns the transcription.

*   **`tts` (Text-to-Speech Service):**
    *   **Technology:** `gTTS` (Google Text-to-Speech), FastAPI
    *   **Role:** Converts text into natural-sounding Bangla speech. Its REST endpoint (`/synthesize`) takes text and returns an MP3 audio stream.

*   **`orchestrator` (The "Brain"):**
    *   **Technology:** Python, `asyncio`, `livekit-sdk`, `aiohttp`
    *   **Role:** The central coordinator. It connects to LiveKit as a bot, listens to the user's audio, and manages the entire conversation flow by calling the VAD, STT, external Government API, and TTS services in sequence.

*   **`frontend` (Web Application):**
    *   **Technology:** HTML, CSS, JavaScript, Nginx
    *   **Role:** A simple, static web page served by Nginx that provides the user interface. It connects to LiveKit, captures microphone audio with a "Push-to-Talk" button, and plays back the bot's audio response.

## Orchestrator Logic Flow

When a user connects and speaks, the following sequence of events is managed by the **Orchestrator**:
1.  **Connect to LiveKit:** The Orchestrator joins the specified LiveKit room as a bot participant.
2.  **Receive Audio:** It subscribes to the user's audio track and starts receiving a real-time stream of audio frames.
3.  **Detect Speech (VAD):** Each small audio chunk is sent to the **VAD Service**.
4.  **Buffer Speech:** When the VAD service detects the start of speech, the Orchestrator begins buffering the incoming audio frames. When the VAD service detects a sufficient period of silence, it marks the end of the user's utterance.
5.  **Transcribe (STT):** The entire buffered audio segment is sent as a single file to the **STT Service**.
6.  **Get Transcription:** The STT service processes the audio on the GPU and returns the transcribed Bangla text.
7.  **Query External API:** The Orchestrator makes a POST request to the **Government Response Service** with the transcribed text. It listens to the streaming response and assembles the complete answer.
8.  **Synthesize (TTS):** The complete answer text is sent to the **TTS Service**.
9.  **Receive Synthesized Audio:** The TTS service returns an MP3 audio stream of the spoken answer.
10. **Stream Response to User:** The Orchestrator decodes the MP3 audio into raw PCM frames and publishes them on its own audio track in the LiveKit room, allowing the user to hear the response in real-time.
