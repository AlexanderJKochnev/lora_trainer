import pandas as pd
import json


def process_beer_data(input_file, output_file):
    """Обработка данных о пиве"""

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
            context_parts.append(f"Стиль: {row['Categories']}")
        if pd.notna(row['Type']):
            context_parts.append(f"Тип: {row['Type']}")
        if pd.notna(row['ABV']):
            context_parts.append(f"Крепость: {row['ABV']}%")
        if pd.notna(row['IBU']):
            context_parts.append(f"Горечь (IBU): {row['IBU']}")

        context = "\n".join(context_parts) if context_parts else "Информация о пиве"

        # Ответ
        output_parts = []
        if pd.notna(row['Tasting Notes']):
            output_parts.append(f"Дегустационные заметки: {row['Tasting Notes']}")
        if pd.notna(row['Description']):
            output_parts.append(f"Описание: {row['Description']}")
        if pd.notna(row['Food Pairing']):
            output_parts.append(f"Сочетание с едой: {row['Food Pairing']}")

        output = "\n".join(output_parts) if output_parts else "Нет описания"

        record = {"instruction": "Опиши характеристики и вкус пива, укажи сочетание с едой", "input": context,
                  "output": output, "metadata": {"source": "beer", "abv": row['ABV'] if pd.notna(row['ABV']) else None,
                                                 "ibu": row['IBU'] if pd.notna(row['IBU']) else None,
                                                 "rating": row['Rating'] if pd.notna(row['Rating']) else None}}
        data.append(record)

        if idx % 2000 == 0:
            print(f"Обработано {idx} записей")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(data)} записей в {output_file}")


if __name__ == "__main__":
    process_beer_data("source/beer_data.csv", "output/beer_lora.jsonl")
