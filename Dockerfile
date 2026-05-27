FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pin PyTorch to CUDA 12.4 wheels (matches Lambda host driver 12.x)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

COPY requirements-train.txt /workspace/requirements-train.txt
RUN python3 -m pip install -r /workspace/requirements-train.txt && \
    (python3 -m pip install flash-attn --no-build-isolation || \
     echo "[docker] flash-attn not installed; training will use sdpa")

COPY . /workspace

ENTRYPOINT ["python3", "-m"]
