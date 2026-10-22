# Frazile AI Image Studio - GPU container
#
# Build:   docker compose build
# Run:     docker compose up
# Open:    http://localhost:7860
#
# Needs the NVIDIA driver on the host and GPU support in Docker
# (Docker Desktop with the WSL 2 backend on Windows, or nvidia-container-toolkit on Linux).

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    FRAZILE_IN_DOCKER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgl1 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.6 first (large layer, cached independently of app changes)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY README.md ./

# Model weights, outputs, the database and the Hugging Face cache live in volumes (see docker-compose.yml)
RUN mkdir -p /app/outputs /app/models/checkpoints /app/models/loras /cache/huggingface

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)" || exit 1

CMD ["python3", "app.py"]
