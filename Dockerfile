# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for model caching
# This tells libraries like Hugging Face, Torch, and Sentence-Transformers
# where to store their downloaded models inside the container.
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install uv and then use it to install packages
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Command to run the application
# Use 0.0.0.0 to expose the port to the network
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
