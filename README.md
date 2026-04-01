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
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

2. # Проверяем версию Docker Compose
docker compose version

# Если версия < 2.0, используйте вариант А (runtime: nvidia)
# Если версия >= 2.0, можно использовать вариант Б