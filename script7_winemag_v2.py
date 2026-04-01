import pandas as pd
import json


def process_winemag_v2(input_file, output_file):
    """Обработка второй версии Wine Magazine данных"""

    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем контекст
        context_parts = []
        if pd.notna(row['winery']):
            context_parts.append(f"Винодельня: {row['winery']}")
        if pd.notna(row['variety']):
            context_parts.append(f"Сорт: {row['variety']}")
        if pd.notna(row['country']):
            context_parts.append(f"Страна: {row['country']}")
        if pd.notna(row['province']):
            context_parts.append(f"Провинция: {row['province']}")
        if pd.notna(row['price']):
            context_parts.append(f"Цена: ${row['price']}")

        context = "\n".join(context_parts) if context_parts else "Информация о вине"

        # Ответ
        description = row['description'] if pd.notna(row['description']) else ""
        points = row['points'] if pd.notna(row['points']) else ""
        title = row['title'] if pd.notna(row['title']) else ""

        output = f"{title}\nОписание: {description}\nРейтинг: {points} баллов"

        record = {"instruction": "Опиши вино и дай профессиональную оценку", "input": context, "output": output,
                  "metadata": {"source": "winemag_v2", "points": points,
                               "taster": row['taster_name'] if pd.notna(row['taster_name']) else None}}
        data.append(record)

        if idx % 20000 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_winemag_v2("source/winemag-data-130k-v2.csv", "output/winemag_v2_lora.jsonl")
