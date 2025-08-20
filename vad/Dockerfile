# Stage 1: Build the application and cache the model
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install build-time dependencies if any (not strictly needed here but good practice)
# RUN apt-get update && apt-get install -y ...

# Install PyTorch for CPU first, as it's a large dependency
# Using the official CPU-only index helps keep the image size smaller.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

# Pre-download the Silero VAD model.
# This command runs the same model loading logic as the app's startup
# to ensure the model is downloaded and cached in this layer.
# The main.py script is configured to use /app/models as the cache dir.
RUN python -c "from main import load_model; load_model()"

# Stage 2: Create the final, smaller production image
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Copy the pre-downloaded model from the builder stage
COPY --from=builder /app/models /app/models

# Copy the application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
