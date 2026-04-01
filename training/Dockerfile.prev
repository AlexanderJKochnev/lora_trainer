FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# Используем devel (для компиляции) и 12.1 (лучше совместимость с torch 2.2.2)

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Теперь создание venv пройдет успешно
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем инструменты сборки внутри venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Сначала Torch
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Затем основные либы
RUN pip install --no-cache-dir \
    transformers==4.45.0 \
    accelerate==0.34.0 \
    peft==0.12.0 \
    datasets==2.21.0 \
    pandas tqdm sentencepiece

# Устанавливаем auto-gptq через pre-built wheel, чтобы избежать ошибки компиляции
RUN pip install --no-cache-dir \
    bitsandbytes==0.43.0 \
    optimum==1.22.0 \
    auto-gptq==0.7.1 --extra-index-url https://github.io

WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
