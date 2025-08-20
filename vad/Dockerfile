# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install PyTorch and Torchaudio for CPU first, as they are large dependencies.
# Using the official CPU-only index helps keep the image size smaller.
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file and install all dependencies, including for testing
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and test code into the container
COPY main.py .
COPY test_main.py .

# Pre-download the Silero VAD model during the build process
# This ensures the model is cached in the image and avoids runtime downloads.
RUN python -c "from main import load_model; load_model()"

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the application for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
