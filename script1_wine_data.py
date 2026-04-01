import pandas as pd
import json
import os


def process_wine_data(input_file, output_file):
    """Обработка wine_data.csv с подробными описаниями вин"""

    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем входной контекст
        context_parts = []
        if pd.notna(row['Name']):
            context_parts.append(f"Название: {row['Name']}")
        if pd.notna(row['Country']):
            context_parts.append(f"Страна: {row['Country']}")
        if pd.notna(row['Brand']):
            context_parts.append(f"Бренд: {row['Brand']}")
        if pd.notna(row['Categories']):
            context_parts.append(f"Категория: {row['Categories']}")
        if pd.notna(row['ABV']):
            context_parts.append(f"Крепость: {row['ABV']}%")
        if pd.notna(row['Food Pairing']):
            context_parts.append(f"Сочетание с едой: {row['Food Pairing']}")
        if pd.notna(row['Sweet-Dry Scale']):
            context_parts.append(f"Сладость: {row['Sweet-Dry Scale']}")
        if pd.notna(row['Body']):
            context_parts.append(f"Тело: {row['Body']}")

        context = "\n".join(context_parts) if context_parts else "Информация о вине"

        # Формируем ответ (дегустационные заметки)
        output_parts = []
        if pd.notna(row['Tasting Notes']):
            output_parts.append(f"Дегустационные заметки: {row['Tasting Notes']}")
        if pd.notna(row['Description']):
            output_parts.append(f"Описание: {row['Description']}")

        output = "\n".join(output_parts) if output_parts else "Нет описания"

        # Создаем запись
        record = {"instruction": "Опиши характеристики и вкус вина на основе предоставленных данных", "input": context,
                  "output": output,
                  "metadata": {"source": "wine_data", "rating": row['Rating'] if pd.notna(row['Rating']) else None,
                               "price": row['Price'] if pd.notna(row['Price']) else None}}
        data.append(record)

        if idx % 5000 == 0:
            print(f"Обработано {idx} записей")

    # Сохраняем в JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")
    return data


if __name__ == "__main__":
    process_wine_data("wine_data.csv", "wine_data_lora.jsonl")
