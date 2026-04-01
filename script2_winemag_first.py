import pandas as pd
import json


def process_wine_magazine(input_file, output_file):
    """Обработка Wine Magazine данных с профессиональными обзорами"""

    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем контекст
        context_parts = []
        if pd.notna(row['winery']):
            context_parts.append(f"Винодельня: {row['winery']}")
        if pd.notna(row['variety']):
            context_parts.append(f"Сорт винограда: {row['variety']}")
        if pd.notna(row['country']):
            context_parts.append(f"Страна: {row['country']}")
        if pd.notna(row['province']):
            context_parts.append(f"Провинция: {row['province']}")
        if pd.notna(row['price']):
            context_parts.append(f"Цена: ${row['price']}")

        context = "\n".join(context_parts) if context_parts else "Информация о вине"

        # Ответ - профессиональное описание
        description = row['description'] if pd.notna(row['description']) else ""
        points = row['points'] if pd.notna(row['points']) else ""

        output = f"Дегустационные заметки: {description}\nРейтинг: {points} баллов"

        record = {"instruction": "Дай профессиональную оценку вину на основе предоставленных характеристик",
                  "input": context, "output": output, "metadata": {"source": "winemag_first150k", "points": points,
                                                                   "variety": row['variety'] if pd.notna(row['variety']) else None}}
        data.append(record)

        if idx % 10000 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_wine_magazine("source/winemag-data_first150k.csv", "output/winemag_lora.jsonl")
