FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch nightly with CUDA 12.4 and sm_120 support
# Try CUDA 12.4 nightly first, then fall back to CUDA 12.3 nightly if not yet published.
RUN python3 -m pip install --pre --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu124 || \
    python3 -m pip install --pre --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu123

RUN python3 -m pip install -r requirements.txt && python3 -m pip install -e .

RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install
RUN cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install

EXPOSE 8080

CMD ["python3", "gradio_app.py", "--host", "0.0.0.0", "--port", "8080"]

