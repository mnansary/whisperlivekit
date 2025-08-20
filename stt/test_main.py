import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io

# Import the app instance from our main script
from main import app

# Create a mock for the Whisper pipeline object
mock_pipeline_object = MagicMock()

def test_health_check_no_gpu():
    """
    Tests the /health endpoint in a simulated environment without a GPU.
    We patch `torch.cuda.is_available` to test this specific scenario.
    """
    with patch('main.torch.cuda.is_available', return_value=False):
        # The lifespan will run on app startup within the TestClient
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            json_response = response.json()
            assert json_response["status"] == "ok"
            # In a no-GPU env, the pipeline should still be loaded (on CPU)
            assert json_response["pipeline_loaded"] is True
            assert json_response["cuda_available"] is False

# We patch the `pipeline` function from `insanely_fast_whisper`
@patch('main.pipeline', return_value=mock_pipeline_object)
def test_transcribe_audio_success(mock_pipeline_func):
    """
    Tests the /transcribe endpoint for a successful case.
    It mocks the transcription pipeline to avoid GPU/model dependency.
    """
    # Configure the mock pipeline to return a sample transcription
    mock_pipeline_object.return_value = {"text": "এটি একটি পরীক্ষা"}

    # Create a fake audio file in memory
    fake_audio_bytes = b"fake_wav_data"
    fake_audio_file = io.BytesIO(fake_audio_bytes)

    with TestClient(app) as client:
        # The 'files' argument for TestClient expects a dictionary
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", fake_audio_file, "audio/wav")}
        )

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["transcription"] == "এটি একটি পরীক্ষা"

        # Verify the mock pipeline was called with the audio content
        mock_pipeline_object.assert_called_once()
        # The first argument of the call is the audio bytes
        assert mock_pipeline_object.call_args[0][0] == fake_audio_bytes

@patch('main.pipeline', return_value=mock_pipeline_object)
def test_transcribe_audio_pipeline_error(mock_pipeline_func):
    """
    Tests how the endpoint handles an exception from the transcription pipeline.
    """
    # Configure the mock pipeline to raise an exception
    mock_pipeline_object.side_effect = Exception("Mocked pipeline error")

    fake_audio_bytes = b"fake_wav_data"
    fake_audio_file = io.BytesIO(fake_audio_bytes)

    with TestClient(app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", fake_audio_file, "audio/wav")}
        )

        assert response.status_code == 500
        assert "Error during transcription" in response.json()["detail"]
