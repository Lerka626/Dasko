import pandas as pd
import numpy as np
from datetime import datetime
import holidays


input_file = 'C:/Users/Lerik/OneDrive/Desktop/all_practices/DaskoWine/data/Чеки Розница для аналитики(2).xlsx'
sheet_name = 'Исходник' 


output_file = 'D:/Download 2/check_for_forecast.xlsx'
ru_holidays = holidays.RU()
df = pd.read_excel(input_file, sheet_name=sheet_name)

columns_needed = [
    'Дата', 'Номер', 'Номенклатура', 'Количество', 'НоменклатураКод',
    'Сумма', 'Себестоимость', 'Магазин', 'Страна', 'Категория',
    'GAP', 'Месяц', 'ГОД', 'ДисконтнаяКарта'
]
df = df[columns_needed].copy()

df['marginCost'] = df['Сумма'] - df['Себестоимость']

df['promoFlag'] = df['ДисконтнаяКарта'].notna().astype(int)

def is_holiday(date):
    if pd.isnull(date):
        return 0
    if isinstance(date, str):
        date = pd.to_datetime(date, errors='coerce')
    if pd.isnull(date):
        return 0
    is_official = date in ru_holidays
    is_late_december = date.month == 12 and date.day >= 18
    return int(is_official or is_late_december)

df['is_holiday_day'] = df['Дата'].apply(is_holiday)

df.to_excel(output_file, index=False)

print(f'Сохранено {len(df)} строк в файл: {output_file}')
