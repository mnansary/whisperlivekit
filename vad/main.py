import os
import torch
import numpy as np
from fastapi import FastAPI, Request, HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# --- Model Loading & Lifespan Management ---
# Use a path that will be in the Docker container for caching.
MODEL_CACHE_DIR = "/app/models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
torch.hub.set_dir(MODEL_CACHE_DIR)

model = None
utils = None

def load_model():
    """Loads the Silero VAD model into the global variables."""
    global model, utils
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True  # Use ONNX for better CPU performance
        )
        logger.info("Silero VAD model loaded successfully (ONNX).")
    except Exception as e:
        logger.error(f"Fatal: Error loading Silero VAD model: {e}")
        # This will prevent the app from starting if the model fails to load.
        raise RuntimeError("Could not load Silero VAD model") from e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    The model is loaded on startup.
    """
    logger.info("Application startup...")
    load_model()
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- VAD Configuration ---
VAD_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz audio
VAD_THRESHOLD = 0.5      # Speech confidence threshold

@app.post("/detect_speech")
async def detect_speech(request: Request):
    """
    Accepts a raw audio chunk (16kHz, 16-bit PCM) and returns whether it contains speech.
    The orchestrator is expected to send audio chunks of a reasonable size (e.g., 256-1536 bytes).
    """
    if not model:
        raise HTTPException(status_code=503, detail="VAD model is not loaded")

    try:
        # Read the raw binary data from the request body
        audio_bytes = await request.body()

        if len(audio_bytes) == 0:
            return {"is_speech": False, "confidence": 0.0}

        # Convert bytes to a NumPy array of int16
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert int16 to float32, which the model expects
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Convert NumPy array to a PyTorch tensor
        audio_tensor = torch.from_numpy(audio_float32)

        # Get speech probability from the model
        speech_prob = model(audio_tensor, VAD_SAMPLE_RATE).item()

        is_speech = speech_prob >= VAD_THRESHOLD

        logger.debug(f"Processed {len(audio_bytes)} bytes. Confidence: {speech_prob:.2f}. Is speech: {is_speech}")

        return {"is_speech": is_speech, "confidence": speech_prob}

    except Exception as e:
        logger.error(f"Error processing audio for VAD: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio")

@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    model_loaded = model is not None
    return {"status": "ok", "model_loaded": model_loaded}
