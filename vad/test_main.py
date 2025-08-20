import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Import the app instance from our main script
from main import app

# Create a mock for the Silero VAD model object.
# This allows us to control its return value in tests.
mock_model_object = MagicMock()

# A fixture to generate a sample audio chunk
@pytest.fixture
def sample_audio_chunk():
    # 16000 samples/sec * 0.030 sec/chunk = 480 samples per chunk
    num_samples = 480
    audio_data = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)
    return audio_data.tobytes()

def test_health_check():
    """
    Tests the /health endpoint. This test runs without mocking to ensure
    the real model can be loaded by the lifespan event handler.
    If the Docker build succeeds (which runs `load_model`), this test should pass.
    """
    # Using TestClient as a context manager ensures lifespan events are triggered
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "ok"
        assert json_response["model_loaded"] is True

# We use a patch on `torch.hub.load` to prevent it from actually downloading
# the model during tests and to control the "model" object it returns.
@patch('main.torch.hub.load', return_value=(mock_model_object, {}))
def test_detect_speech_when_speech_is_present(mock_hub_load, sample_audio_chunk):
    """
    Tests the /detect_speech endpoint when the mocked model detects speech.
    """
    # Configure the mock model to return a high confidence score
    mock_model_object.return_value.item.return_value = 0.8

    # The patch is active within this function's scope.
    # The TestClient will trigger the lifespan startup, which calls `load_model`.
    # `load_model` will call our patched `torch.hub.load`.
    with TestClient(app) as client:
        response = client.post("/detect_speech", content=sample_audio_chunk)

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["is_speech"] is True
        assert json_response["confidence"] == 0.8

@patch('main.torch.hub.load', return_value=(mock_model_object, {}))
def test_detect_speech_when_speech_is_absent(mock_hub_load, sample_audio_chunk):
    """
    Tests the /detect_speech endpoint when the mocked model does not detect speech.
    """
    # Configure the mock model to return a low confidence score
    mock_model_object.return_value.item.return_value = 0.2

    with TestClient(app) as client:
        response = client.post("/detect_speech", content=sample_audio_chunk)

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["is_speech"] is False
        assert json_response["confidence"] == 0.2

@patch('main.torch.hub.load', return_value=(mock_model_object, {}))
def test_detect_speech_with_empty_payload(mock_hub_load):
    """
    Tests the /detect_speech endpoint with an empty request body.
    """
    with TestClient(app) as client:
        response = client.post("/detect_speech", content=b"")

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["is_speech"] is False
        assert json_response["confidence"] == 0.0
