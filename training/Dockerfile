FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка Unsloth и зависимостей одной командой
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com" \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    transformers==4.45.0 accelerate==0.34.0 peft==0.12.0 datasets==2.21.0 \
    bitsandbytes==0.43.3 xformers==0.0.27.post2 trl==0.9.6

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
