# Use official PyTorch GPU accelerated base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PAGER=cat

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install vg (variation graph) toolkit static binary
# Using v1.55.0 release
RUN wget -q https://github.com/vgteam/vg/releases/download/v1.55.0/vg && \
    chmod +x vg && \
    mv vg /usr/local/bin/

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose port for FastAPI backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
