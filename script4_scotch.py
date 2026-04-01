import pandas as pd
import json


def process_scotch_reviews(input_file, output_file):
    """Обработка отзывов о шотландском виски"""

    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем контекст
        context_parts = []
        if pd.notna(row['name']):
            context_parts.append(f"Название: {row['name']}")
        if pd.notna(row['category']):
            context_parts.append(f"Категория: {row['category']}")
        if pd.notna(row['price']):
            context_parts.append(f"Цена: {row['price']}")

        context = "\n".join(context_parts) if context_parts else "Информация о виски"

        # Ответ
        output_parts = []
        if pd.notna(row['description']):
            output_parts.append(f"Описание: {row['description']}")
        if pd.notna(row['review.point']):
            output_parts.append(f"Оценка: {row['review.point']} баллов")

        output = "\n".join(output_parts) if output_parts else "Нет описания"

        record = {"instruction": "Оцени качество и опиши вкус шотландского виски", "input": context, "output": output,
                  "metadata": {"source": "scotch_review",
                               "points": row['review.point'] if pd.notna(row['review.point']) else None}}
        data.append(record)

        if idx % 500 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_scotch_reviews("source/scotch_review.csv", "output/scotch_lora.jsonl")
