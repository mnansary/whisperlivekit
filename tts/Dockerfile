# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies, including for testing
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and test code into the container
COPY main.py .
COPY test_main.py .

# Expose the port the app runs on
EXPOSE 8002

# Default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
