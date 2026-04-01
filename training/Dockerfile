# training/Dockerfile
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Используем версии, совместимые с Qwen2.5-7B-GPTQ и Python 3.10
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    peft==0.8.2 \
    datasets==2.16.1 \
    bitsandbytes==0.42.0 \
    auto-gptq==0.6.0 \
    optimum==1.17.1 \
    pandas \
    tqdm \
    sentencepiece

WORKDIR /workspace

COPY train_lora.py .

ENTRYPOINT ["python3", "train_lora.py"]