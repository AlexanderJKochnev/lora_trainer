import json
import pandas as pd


def analyze_dataset(file_path):
    """Анализирует сформированный датасет"""

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    print(f"=== Анализ датасета {file_path} ===")
    print(f"Всего записей: {len(df)}")
    print(f"\nДлина instruction:")
    print(df['instruction'].str.len().describe())

    print(f"\nДлина input:")
    print(df['input'].str.len().describe())

    print(f"\nДлина output:")
    print(df['output'].str.len().describe())

    print(f"\nРаспределение по источникам:")
    sources = df['metadata'].apply(lambda x: x.get('source', 'unknown'))
    print(sources.value_counts())

    # Проверка на пустые поля
    empty_input = df[df['input'].str.len() < 10]
    print(f"\nЗаписей с коротким input (<10 символов): {len(empty_input)}")

    empty_output = df[df['output'].str.len() < 10]
    print(f"Записей с коротким output (<10 символов): {len(empty_output)}")

    return df


if __name__ == "__main__":
    analyze_dataset("output/sommelier_lora_full.jsonl")
