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

# 2. Устанавливаем ВСЕ зависимости ЖЕСТКО (чтобы Unsloth их не обновлял)
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    datasets==2.19.2 \
    accelerate==0.30.1 \
    peft==0.11.1 \
    trl==0.8.6 \
    bitsandbytes==0.43.1 \
    xformers==0.0.26.post1 \
    sentencepiece \
    click==8.1.7 \
    numpy==1.26.4

# 3. Устанавливаем Unsloth из Git БЕЗ проверки зависимостей (--no-deps)
# Это займет 10-20 секунд, так как компиляция не потребуется
RUN pip install --no-cache-dir --no-deps \
    "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
