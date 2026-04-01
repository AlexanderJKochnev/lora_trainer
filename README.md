# lora_trainer
trainer for lora adapter
list of scripts
python script1_wine_data.py
python script2_winemag_first.py
python script3_spirits.py
python script4_scotch.py
python script5_beer.py
python script6_wine.py
python script7_winemag_v2.py
python script8_merge.py
python script9_analysis.py

1. # Проверяем, что Docker может использовать GPU
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

2. # Проверяем версию Docker Compose
docker compose version

3. # Проверяем, что датасет существует
ls -la ./output/sommelier_lora_full.jsonl

4. # Проверяем, что модель существует
ls -la /mnt/hdd_data/volumes/vllm_node/models/Qwen2.5-7B-GPTQ/

5. # Проверяем директорию для адаптеров
ls -la /mnt/hdd_data/volumes/vllm_node/lora_adapters/

6. # Запуск обучения (2-8 часов в зависимости от разммера базы)
docker compose -f up

7. # мониторинг обучения (в другом терминале)
watch -n 1 nvidia-smi

8. # Проверяем созданный адаптер
ls -la /mnt/hdd_data/volumes/vllm_node/lora_adapters/sommelier_lora_v1/

# Должны увидеть:
# adapter_config.json
# adapter_model.bin

9. # Останавливаем vLLM
docker stop vllm_server

10. # Запускаем vLLM с новым адаптером
docker run -d \
  --name vllm_server \
  --runtime nvidia \
  -v /mnt/hdd_data/volumes/vllm_node/models/Qwen2.5-7B-GPTQ:/model \
  -v /mnt/hdd_data/volumes/vllm_node/lora_adapters:/app/lora \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --quantization gptq \
  --enable-lora \
  --lora-modules sommelier=/app/lora/sommelier_lora_v1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code