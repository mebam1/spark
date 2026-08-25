FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION="2.4.1"
ARG TORCHVISION_VERSION="0.19.1"
ARG TORCH_CLUSTER_VERSION="1.6.3"
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9"
ARG MAX_JOBS="2"

ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYOPENGL_PLATFORM=egl \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    ninja-build \
    pkg-config \
    python-is-python3 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv

WORKDIR /workspace/SPARK

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cu124 \
    && pip install torch-cluster==${TORCH_CLUSTER_VERSION} -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu124.html \
    && grep -vE '^(torch|torchvision|torch-cluster)==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && pip install ninja \
    && pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

CMD ["bash"]
