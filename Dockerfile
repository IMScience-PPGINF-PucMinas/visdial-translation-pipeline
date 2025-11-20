FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY translate_visdial.py .

# Create directories for data
RUN mkdir -p /data/input /data/output /data/cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/data/cache
ENV HF_HOME=/data/cache

# Default command
ENTRYPOINT ["python", "translate_visdial.py"]
CMD ["--help"]