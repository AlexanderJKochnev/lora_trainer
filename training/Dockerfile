FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# Используем devel (для компиляции) и 12.1 (лучше совместимость с torch 2.2.2)

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip и устанавливаем wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Сначала Torch
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://pytorch.org

# Затем основные либы
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    peft==0.8.2 \
    datasets==2.16.1 \
    pandas \
    tqdm \
    sentencepiece

# Устанавливаем auto-gptq через pre-built wheel, чтобы избежать ошибки компиляции
RUN pip install --no-cache-dir \
    bitsandbytes==0.42.0 \
    optimum==1.17.1 \
    auto-gptq==0.6.0 --extra-index-url https://github.io

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
