
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import os

# Конфигурация
DB_PATH = r"Z:\TF_GPU\sales_data.db"
OUTPUT_CSV = r"Z:\TF_GPU\full_series_analysis.csv"

conn = sqlite3.connect(DB_PATH)

# Получаем все уникальные комбинации магазин + товар
query_items = "SELECT DISTINCT Магазин, НоменклатураКод FROM sales_data"
items = pd.read_sql_query(query_items, conn)

results = []

for _, row in items.iterrows():
    store = row['Магазин']
    item_code = row['НоменклатураКод']
    
    # Загружаем данные для этой комбинации
    query_data = f"""
    SELECT Дата, Количество
    FROM sales_data
    WHERE Магазин = '{store}' AND НоменклатураКод = '{item_code}'
    ORDER BY Дата
    """
    df = pd.read_sql_query(query_data, conn)
    
    if df.empty:
        continue
    
    df['Дата'] = pd.to_datetime(df['Дата'])
    df = df.set_index('Дата').asfreq('D', fill_value=0)  # Заполняем пропуски нулями
    
    values = df['Количество'].values
    
    # Полнота и выбросы
    missing_ratio = df['Количество'].isna().mean()
    z = np.abs(zscore(values))
    outliers_ratio = (z > 3).mean()
    
    # Стационарность
    try:
        adf_stat, adf_pvalue, *_ = adfuller(values)
    except:
        adf_stat, adf_pvalue = np.nan, np.nan
    
    # Декомпозиция тренда и сезонности
    try:
        decomp = seasonal_decompose(values, model='additive', period=7, extrapolate_trend='freq')
        trend_strength = np.nan if decomp.trend is None else np.nanvar(decomp.trend) / np.nanvar(values)
        seasonal_strength = np.nan if decomp.seasonal is None else np.nanvar(decomp.seasonal) / np.nanvar(values)
    except:
        trend_strength, seasonal_strength = np.nan, np.nan
    
    results.append({
        "Магазин": store,
        "НоменклатураКод": item_code,
        "Длина_ряда": len(values),
        "Доля_пропусков": missing_ratio,
        "Доля_выбросов": outliers_ratio,
        "ADF_статистика": adf_stat,
        "ADF_pvalue": adf_pvalue,
        "Сила_тренда": trend_strength,
        "Сила_сезонности": seasonal_strength
    })

conn.close()

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Анализ завершен, отчёт сохранён: {OUTPUT_CSV}")
