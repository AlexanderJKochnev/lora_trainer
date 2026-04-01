import json
import os
from pathlib import Path


def merge_all_datasets(input_files, output_file, sample_size=None):
    """
    Объединяет все JSONL файлы в один датасет
    sample_size: если указан, берет только N записей из каждого файла
    """

    all_data = []

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            continue

        print(f"Чтение {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if sample_size:
                lines = lines[:sample_size]

            for line in lines:
                try:
                    record = json.loads(line)
                    all_data.append(record)
                except json.JSONDecodeError:
                    print(f"Ошибка парсинга в {file_path}")
                    continue

        print(f"  Добавлено {len(lines)} записей")

    print(f"\nВсего записей: {len(all_data)}")

    # Сохраняем объединенный файл
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено в {output_file}")

    # Создаем статистику по источникам
    sources = {}
    for record in all_data:
        source = record.get('metadata', {}).get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print("\nСтатистика по источникам:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} записей")


if __name__ == "__main__":
    # Список всех сгенерированных файлов
    input_files = ["output/wine_data_lora.jsonl",
                   "output/winemag_lora.jsonl",
                   "output/spirits_lora.jsonl",
                   "output/scotch_lora.jsonl",
                   "output/beer_lora.jsonl",
                   "output/wine_lora.jsonl",
                   "output/winemag_v2_lora.jsonl"]

    # Для теста можно взять только первые 1000 записей из каждого файла
    # merge_all_datasets(input_files, "sommelier_lora_test.jsonl", sample_size=1000)

    # Полный датасет
    merge_all_datasets(input_files, "output/sommelier_lora_full.jsonl")
