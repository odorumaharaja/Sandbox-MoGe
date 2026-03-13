# syntax=docker/dockerfile:1

# ==========================================
# Build Arguments
# ==========================================
ARG CUDA_VERSION=12.9.1
ARG OS_VERSION=ubuntu22.04
ARG TAG_VERSION=${CUDA_VERSION}-cudnn-runtime-${OS_VERSION}

# ==========================================
# Base Environment Setup Stage
# ==========================================
FROM nvidia/cuda:${TAG_VERSION}

# Get uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent interactive mode during installation
ENV DEBIAN_FRONTEND=noninteractive
# Do not buffer Python output (log immediately)
ENV PYTHONUNBUFFERED=1

# Install system packages
# git and image processing libraries (libgl1, libglib2.0-0)
# python3 and pip are managed by uv, so they are excluded from system packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set directory for Hugging Face model cache
ENV HF_HOME=/app/hf_cache

# uv configuration
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy application code
COPY *.py /app
COPY .python-version /app
COPY pyproject.toml /app

# Sync project (install after copying code)
RUN uv sync && uv cache clean

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose Gradio port
EXPOSE 7860

# Run application
CMD ["uv", "run", "app.py"]
