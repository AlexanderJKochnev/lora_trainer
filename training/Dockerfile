FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. Сначала ставим БАЗУ (Torch 2.3.0)
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. СТАВИМ UNSLOTH ИЗ PYPI (НЕ ИЗ GITHUB)
# Это скачает готовую сборку вместо компиляции исходников
RUN pip install --no-cache-dir \
    unsloth==2024.5 \
    xformers==0.0.26.post1 \
    trl==0.8.6 \
    peft==0.11.1 \
    accelerate==0.30.1 \
    transformers==4.41.2 \
    bitsandbytes==0.43.1 \
    datasets==2.19.2

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
