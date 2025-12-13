import pandas as pd

# Загрузка данных
df = pd.read_csv("D:/TF_GPU/full_series_analysis.csv")

# Обезличивание
df["Магазин_ID"] = (
    df["Магазин"]
    .astype("category")
    .cat.codes
    .add(1)
    .apply(lambda x: f"Store_{x:03d}")
)

df["Номенклатура_ID"] = (
    df["НоменклатураКод"]
    .astype("category")
    .cat.codes
    .add(1)
    .apply(lambda x: f"Item_{x:05d}")
)

# Подсчёт уникальных значений (для имени файла)
n_stores = df["Магазин_ID"].nunique()
n_items = df["Номенклатура_ID"].nunique()

# Формирование имени файла
output_filename = f"anonymized_Stores_{n_stores}_Items_{n_items}.csv"

# Удаление исходных колонок
df = df.drop(columns=["Магазин", "НоменклатураКод"])

# Сохранение
df.to_csv(output_filename, index=False)

print(f"Файл сохранён: {output_filename}")
