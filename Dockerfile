FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Install pip and upgrade it
RUN python -m pip install --upgrade pip

# Install PyTorch and other dependencies
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install spconv v2.x with CUDA 11.6
RUN pip install spconv-cu116

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install the LION codebase
RUN pip install -e . 