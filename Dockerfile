# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install native system dependencies required by OpenCV (used by EasyOCR)
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for model caching
# This tells libraries like Hugging Face, Torch, and Sentence-Transformers
# where to store their downloaded models inside the container.
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install uv and then use it to install packages
# BEST PRACTICE: Install CPU-only PyTorch first to avoid downloading 4.5GB of unused CUDA GPU binaries
RUN pip install uv && \
    uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make the entrypoint script executable and fix Windows line endings
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Expose port 8000 for Azure App Service to detect
EXPOSE 8000

# Command to run the application using the entrypoint script
CMD ["./entrypoint.sh"]
