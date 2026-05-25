# Use lightweight stable Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (git is needed for some HF components)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU-only version to save massive amounts of container space (~2GB reduction)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies manifest
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server implementation files
COPY server.py .
COPY download_model.py .

# Expose FastAPI default port
EXPOSE 8000

# Run the server on startup
# It will check for the model in the cache (mounted via volume) or download it automatically
CMD ["python", "server.py"]
