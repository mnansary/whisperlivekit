import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Import the app instance from our main script
from main import app

# Create a client for the tests
client = TestClient(app)

def test_health_check():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "TTS"}

# We patch the gTTS class to avoid making real network calls to Google's services
@patch('main.gTTS')
def test_synthesize_speech_success(mock_gtts_class):
    """
    Tests the /synthesize endpoint for a successful case.
    It mocks the gTTS library to ensure the API handles the logic correctly.
    """
    # Configure the mock gTTS instance that will be created
    mock_instance = mock_gtts_class.return_value

    # Sample audio data to be "generated" by the mock
    fake_audio_bytes = b'fake_mp3_data'

    # The write_to_fp method needs to write our fake data to the stream
    def write_to_fp_mock(fp):
        fp.write(fake_audio_bytes)

    mock_instance.write_to_fp = write_to_fp_mock

    # Make the request to the endpoint
    response = client.post("/synthesize", json={"text": "হ্যালো বিশ্ব", "lang": "bn"})

    # Assertions
    assert response.status_code == 200
    assert response.headers['content-type'] == 'audio/mpeg'
    assert response.content == fake_audio_bytes

    # Verify that gTTS was called with the correct parameters
    mock_gtts_class.assert_called_once_with(text='হ্যালো বিশ্ব', lang='bn', slow=False)

def test_synthesize_speech_empty_text():
    """
    Tests that the /synthesize endpoint returns a 400 Bad Request
    if the input text is empty or just whitespace.
    """
    response = client.post("/synthesize", json={"text": " "})
    assert response.status_code == 400
    assert response.json() == {"detail": "Input text cannot be empty."}

@patch('main.gTTS')
def test_synthesize_speech_gtts_error(mock_gtts_class):
    """
    Tests how the endpoint handles an error from the gTTS library.
    """
    # Import the specific error class from the library
    from gtts import gTTSError

    # Configure the mock to raise a gTTSError
    mock_gtts_class.side_effect = gTTSError("Mocked gTTS error")

    response = client.post("/synthesize", json={"text": "test"})

    assert response.status_code == 500
    assert "Failed to synthesize speech" in response.json()["detail"]
