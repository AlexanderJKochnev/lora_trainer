import pandas as pd
import json


def process_spirits_data(input_file, output_file):
    """Обработка данных по крепкому алкоголю"""

    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем контекст
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
        if pd.notna(row['Base Ingredient']):
            context_parts.append(f"Основа: {row['Base Ingredient']}")
        if pd.notna(row['Years Aged']):
            context_parts.append(f"Выдержка: {row['Years Aged']} лет")

        context = "\n".join(context_parts) if context_parts else "Информация о напитке"

        # Ответ - дегустационные заметки
        output_parts = []
        if pd.notna(row['Tasting Notes']):
            output_parts.append(f"Дегустационные заметки: {row['Tasting Notes']}")
        if pd.notna(row['Description']):
            output_parts.append(f"Описание: {row['Description']}")

        output = "\n".join(output_parts) if output_parts else "Нет описания"

        record = {"instruction": "Опиши характеристики и вкус крепкого алкогольного напитка", "input": context,
                  "output": output, "metadata": {"source": "spirits_data",
                                                 "category": row['Categories'] if pd.notna(row['Categories']) else None,
                                                 "rating": row['Rating'] if pd.notna(row['Rating']) else None}}
        data.append(record)

        if idx % 2000 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_spirits_data("source/spirits_data.csv", "output/spirits_lora.jsonl")
