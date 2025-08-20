# Use an NVIDIA CUDA base image compatible with RTX 3090 and PyTorch
# Using a devel image provides headers needed for building some python packages
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during build
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python, pip, and ffmpeg (for audio processing)
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install the CUDA-enabled version of PyTorch and its ecosystem
# This version is compatible with the CUDA 11.8 in the base image
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the rest of the application's Python dependencies, including for testing
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application and test code into the container
COPY main.py .
COPY test_main.py .

# Expose the port the app will run on
EXPOSE 8001

# Command to run the application using uvicorn
# Using 0.0.0.0 makes the service accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
