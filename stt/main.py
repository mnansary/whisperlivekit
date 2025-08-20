import os
import torch
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from insanely_fast_whisper import pipeline
import pydub

# --- Configuration ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8  # Adjust based on available VRAM
# Use float16 for modern GPUs like the 3090 for better performance
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else torch.float32

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
pipe = None

@app.on_event("startup")
def load_pipeline():
    """Load the transcription pipeline on application startup."""
    global pipe
    if not torch.cuda.is_available():
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.warning("!!! CUDA not available. Running on CPU will be slow. !!!")
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        logger.info(f"Loading model '{MODEL_NAME}' on device '{DEVICE}' with dtype '{TORCH_DTYPE}'...")
        # The pipeline function from insanely-fast-whisper handles the model loading
        # and optimization with ctranslate2 under the hood.
        pipe = pipeline(
            model=MODEL_NAME,
            device=DEVICE,
            torch_dtype=TORCH_DTYPE,
        )
        logger.info("Transcription pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal: Error loading transcription pipeline: {e}")
        raise RuntimeError("Could not load transcription pipeline") from e

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes it using the Whisper model, and returns the text.
    """
    if not pipe:
        raise HTTPException(status_code=503, detail="Transcription pipeline is not loaded.")

    try:
        # Read the content of the uploaded file into memory
        audio_content = await file.read()
        logger.info(f"Received audio file: {file.filename} ({len(audio_content)} bytes)")

        # The pipeline can handle raw bytes directly. It's robust to different formats.
        logger.info("Starting transcription...")
        transcription_result = pipe(
            audio_content,
            batch_size=BATCH_SIZE,
            generate_kwargs={
                "language": "bengali",  # Specify "bengali" for Bangla language
                "task": "transcribe",
            },
        )

        transcribed_text = transcription_result["text"].strip()
        logger.info(f"Transcription successful. Text: '{transcribed_text}'")

        return JSONResponse(content={"transcription": transcribed_text})

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify the service and model status."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    return {
        "status": "ok",
        "pipeline_loaded": pipe is not None,
        "cuda_available": cuda_available,
        "device": DEVICE,
        "device_name": device_name,
        "model": MODEL_NAME,
        "torch_dtype": str(TORCH_DTYPE),
    }
