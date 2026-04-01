import pandas as pd
import json


def process_wine_dataset(input_file, output_file):
    """Обработка большого датасета вин"""

    # Читаем с оптимизацией для большого файла
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Загружено {len(df)} записей из {input_file}")

    data = []

    for idx, row in df.iterrows():
        # Формируем контекст
        context_parts = []
        if pd.notna(row['wine']):
            context_parts.append(f"Вино: {row['wine']}")
        if pd.notna(row['winery']):
            context_parts.append(f"Винодельня: {row['winery']}")
        if pd.notna(row['varietal']):
            context_parts.append(f"Сорт: {row['varietal']}")
        if pd.notna(row['appellation']):
            context_parts.append(f"Апелласьон: {row['appellation']}")
        if pd.notna(row['category']):
            context_parts.append(f"Категория: {row['category']}")
        if pd.notna(row['designation']):
            context_parts.append(f"Дизайн/Линейка: {row['designation']}")
        if pd.notna(row['alcohol']):
            context_parts.append(f"Алкоголь: {row['alcohol']}%")

        context = "\n".join(context_parts) if context_parts else "Информация о вине"

        # Ответ
        output_parts = []
        if pd.notna(row['review']):
            output_parts.append(f"Обзор: {row['review']}")

        output = "\n".join(output_parts) if output_parts else "Нет описания"

        record = {"instruction": "Дай экспертную оценку вину на основе его характеристик", "input": context,
                  "output": output,
                  "metadata": {"source": "wine", "rating": row['rating'] if pd.notna(row['rating']) else None,
                               "price": row['price'] if pd.notna(row['price']) else None,
                               "reviewer": row['reviewer'] if pd.notna(row['reviewer']) else None}}
        data.append(record)

        if idx % 50000 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_wine_dataset("source/wine.csv", "output/wine_lora.jsonl")
