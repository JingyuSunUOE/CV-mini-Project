FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common nfs-common\
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
    build-essential \
    nano \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3.10 -m uv pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    invisible_watermark && \
    python3.10 -m pip install --no-cache-dir \
    accelerate \
    datasets \
    opencv-python \
    hf-doc-builder \
    hf-transfer \
    huggingface-hub \
    Jinja2 \
    librosa \
    numpy \
    scipy \
    tensorboard \
    transformers \
    pytorch-lightning \
    diffusers[torch] \
    av==12.2.0 \
    lightly==1.5.8 \
    lightly-utils==0.0.2 \
    lightning-utilities==0.11.3.post0 \
    pytorch-lightning==2.3.1 \
    timm==1.0.7 \
    tqdm==4.66.2 \
    wandb==0.17.4 \
    numpy>=1.18.1 \
    matplotlib==3.9.1 \
    nvitop \
    kornia \
    monai \
    ftfy \
    regex \
    clip

# install Java
RUN apt-get update && apt-get install -y default-jdk

WORKDIR /mnt/ceph_rbd/users/jingyu/cv_assignment/

# Create a directory to store output
RUN mkdir -p /output
