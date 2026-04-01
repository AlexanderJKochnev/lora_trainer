# training/Dockerfile
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.1.0-base-ubuntu22.04
FROM nvidia/cuda:12.6.0-base-ubuntu24.04
# Устанавливаем Python
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем пакеты (с поддержкой GPTQ)
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
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

COPY train_lora.py .

ENTRYPOINT ["python3", "train_lora.py"]