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

# 2. Устанавливаем зависимости с ЖЕСТКОЙ фиксацией NVIDIA-библиотек
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvtx-cu12==12.1.105 \
    triton==2.3.0 \
    transformers==4.41.2 \
    datasets==2.19.2 \
    accelerate==0.30.1 \
    peft==0.11.1 \
    trl==0.8.6 \
    bitsandbytes==0.43.1 \
    xformers==0.0.26.post1 \
    unsloth_zoo \
    sentencepiece


# 3. Устанавливаем Unsloth из Git БЕЗ проверки зависимостей (--no-deps)
# Это займет 10-20 секунд, так как компиляция не потребуется
RUN pip install --no-cache-dir --no-deps \
    "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
