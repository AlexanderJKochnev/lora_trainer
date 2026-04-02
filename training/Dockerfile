FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. СНАЧАЛА ставим NVIDIA-зависимости (балласт), чтобы torch их не искал
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
    nvidia-nvtx-cu12==12.1.105


# СЛОЙ 2: ТЯЖЕЛЫЙ TORCH (Замораживаем 800 Мб)
RUN pip install --no-cache-dir --no-deps \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Устанавливаем Xformers отдельно (он тяжелый и капризный)
RUN pip install --no-cache-dir --no-deps \
    xformers==0.0.26.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# СЛОЙ 3: СТАБИЛЬНЫЕ ML-БИБЛИОТЕКИ (Чтобы pip не "думал")
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    datasets==2.19.2 \
    accelerate==0.30.1 \
    peft==0.11.1 \
    trl==0.8.6 \
    bitsandbytes==0.43.1 \
    sentencepiece

RUN pip install --no-cache-dir --no-deps unsloth_zoo

# СЛОЙ 4: ФИНАЛЬНЫЙ UNSLOTH (Ставим без зависимостей, чтобы ничего не сломать)
RUN pip install --no-cache-dir --no-deps \
    "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# ЭТАП 2: FINAL (Чистый образ без мусора сборки)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Устанавливаем только минимальный Python для запуска
RUN apt-get update && apt-get install -y python3.10 python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Копируем ГОТОВОЕ окружение из билдера
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


WORKDIR /workspace
COPY train_lora.py .
ENTRYPOINT ["python3", "train_lora.py"]
