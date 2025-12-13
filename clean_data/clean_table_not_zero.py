import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3

# Конфигурация
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sales_path = os.path.join(BASE_DIR, "data", "check_for_forecast.xlsx")
nomenclature_path = os.path.join(BASE_DIR, "data", "Заполненность1.xlsx")
output_dir = os.path.join(BASE_DIR, 'training_data')
db_path = os.path.join(BASE_DIR, 'sales_data.db')

# Маппинг GAP из справочника к общим категориям
GAP_MAPPING = {
    "Вино/винные напитки": ["Вино", "Столовое вино", "Креплёное вино", "Вермут", "Газированное вино", "Плодово-фруктовое вино"],
    "Вино игристое": ["Вино игристое"],
    "ЛВИ": ["Виски", "Джин", "Ликер", "Ром", "Дистиллят", "Настойка", "Саке", "Аквавит"],
    "Слабоалкогольные напитки": ["Пиво", "Прочее"],
    "Водка": ["Водка"],
    "Бренди": ["Бренди"]
}

def load_and_preprocess_data():
    """Загрузка и предварительная обработка данных"""
    
    # Загрузка данных о продажах
    sales_df = pd.read_excel(sales_path)
    print(f"Размер sales_df до обработки: {sales_df.shape}")
    
    # Удаляем ненужные столбцы (цены и дополнительные категории)
    columns_to_drop = ['Номер', 'ДисконтнаяКарта', 'Сумма', 'Себестоимость', 'marginCost', 'Вид', 'Подкатегория']
    for col in columns_to_drop:
        if col in sales_df.columns:
            sales_df = sales_df.drop(columns=[col])
    
    print(f"Размер sales_df после удаления столбцов: {sales_df.shape}")
    
    # Предобработка названий столбцов
    sales_df.columns = sales_df.columns.str.strip()
    
    # Загрузка справочника номенклатуры
    nomenclature_df = pd.read_excel(nomenclature_path)
    nomenclature_df.columns = nomenclature_df.columns.str.strip()
    
    # Переименовываем столбцы для удобства
    nomenclature_df = nomenclature_df.rename(columns={
        'Код': 'НоменклатураКод', 
        'ГАП': 'GAP',
        'Наименование': 'Номенклатура'
    })
    
    return sales_df, nomenclature_df

def aggregate_sales_data(df):
    """Схлопывание таблицы продаж - только фактические продажи"""
    print("Начинаем схлопывание таблицы...")
    
    # Группируем по ключевым полям
    grouped = df.groupby(['Дата', 'Магазин', 'Номенклатура', 'НоменклатураКод'])
    
    # Агрегируем данные - только фактические продажи
    aggregated = grouped.agg({
        'Количество': 'sum',
        'promoFlag': 'max',
        'is_holiday_day': 'first'
    }).reset_index()
    
    # Добавляем недостающие столбцы
    for col in ['Страна', 'Категория', 'GAP', 'Месяц', 'ГОД']:
        if col in df.columns:
            aggregated[col] = grouped[col].first().values
    
    print(f"Размер после схлопывания: {aggregated.shape}")
    return aggregated

def map_gap_to_category(gap):
    """Маппинг GAP к категории"""
    for category, gaps in GAP_MAPPING.items():
        if gap in gaps:
            return category
    return gap  # Если не нашли маппинг, возвращаем исходный GAP

def merge_with_nomenclature(sales_df, nomenclature_df):
    """Слияние данных продаж со справочником номенклатуры"""
    print("Слияние данных со справочником номенклатуры...")
    
    # Создаем копии данных для безопасного изменения
    sales_data = sales_df.copy()
    nom_data = nomenclature_df.copy()
    
    # Обеспечиваем совместимость типов данных для ключевых столбцов
    for df in [sales_data, nom_data]:
        if 'НоменклатураКод' in df.columns:
            df['НоменклатураКод'] = df['НоменклатураКод'].astype(str)
        if 'Номенклатура' in df.columns:
            df['Номенклатура'] = df['Номенклатура'].astype(str)
    
    # Создаем словарь для быстрого доступа к данным справочника
    nom_dict = {}
    for _, row in nom_data.iterrows():
        key = (str(row['НоменклатураКод']), str(row['Номенклатура']))
        nom_dict[key] = {
            'GAP': row['GAP'],
            'Категория': row['Категория'],
            'Страна': row.get('Страна', None)
        }
    
    # Обновляем данные продаж на основе справочника
    for idx, row in sales_data.iterrows():
        key = (str(row['НоменклатураКод']), str(row['Номенклатура']))
        
        if key in nom_dict:
            nom_data = nom_dict[key]
            
            # Обновляем GAP и Категорию из справочника
            if pd.notna(nom_data['GAP']):
                sales_data.at[idx, 'GAP'] = nom_data['GAP']
            if pd.notna(nom_data['Категория']):
                sales_data.at[idx, 'Категория'] = nom_data['Категория']
            
            # Обновляем дополнительные поля из справочника
            for col in ['Страна']:
                if nom_data[col] is not None and pd.notna(nom_data[col]):
                    sales_data.at[idx, col] = nom_data[col]
    
    return sales_data

def determine_gap_for_remaining(sales_df, nomenclature_df):
    """Определение GAP для товаров, отсутствующих в справочнике"""
    print("Определение GAP для оставшихся товаров...")
    
    # Создаем копию данных
    enhanced_df = sales_df.copy()
    
    # Создаем список GAP из справочника
    known_gaps = set(nomenclature_df['GAP'].dropna().unique())
    
    # Список алкогольных ключевых слов для более точного определения
    alcohol_keywords = [
        'вино', 'водка', 'пиво', 'коньяк', 'виски', 'джин', 'ром', 'эль', 'бренди',
        'шампанское', 'ликер', 'текила', 'саке', 'сидр', 'медовуха', 'вермут',
        'абсент', 'граппа', 'наливка', 'настойка', 'портвейн', 'херес', 'мадера', 'агатат-голд',
        'мартини', 'портвейн', 'баррель', 'пивной напиток', 'слабоалкогольный напиток', 'санто стефано',
    ]
    
    # Для товаров, у которых GAP не установлен или не соответствует известным GAP
    for idx, row in enhanced_df.iterrows():
        if pd.isna(row['GAP']) or row['GAP'] not in known_gaps:
            # Более точная проверка на алкогольные товары
            is_alcohol = False
            category = str(row['Категория']).lower() if pd.notna(row['Категория']) else ""
            nomenclature = str(row['Номенклатура']).lower() if pd.notna(row['Номенклатура']) else ""
            
            # Проверяем по ключевым словам в категории и наименовании
            for keyword in alcohol_keywords:
                if keyword in category.lower() or keyword in nomenclature.lower():
                    is_alcohol = True
                    break
            
            if is_alcohol:
                enhanced_df.at[idx, 'GAP'] = 'Другие'
            else:
                enhanced_df.at[idx, 'GAP'] = 'Не напиток'
    
    return enhanced_df

def remove_nomenclature_column(df):
    """Удаление столбца с наименованием товара перед сохранением"""
    print("Удаление столбца 'Номенклатура'...")
    if 'Номенклатура' in df.columns:
        df = df.drop(columns=['Номенклатура'])
        print("Столбец 'Номенклатура' удален")
    return df

def setup_database(db_path):
    """Настройка базы данных и создание таблиц"""
    conn = sqlite3.connect(db_path)
    
    # Создаем основную таблицу для хранения всех данных (БЕЗ Номенклатуры)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS sales_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Дата TEXT,
        Магазин TEXT,
        НоменклатураКод TEXT,
        Страна TEXT,
        Категория TEXT,
        GAP TEXT,
        Количество REAL,
        promoFlag INTEGER,
        Месяц INTEGER,
        ГОД INTEGER,
        is_holiday_day INTEGER
    )
    ''')
    
    # Создаем индексы для ускорения запросов
    conn.execute('CREATE INDEX IF NOT EXISTS idx_shop_gap ON sales_data (Магазин, GAP)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON sales_data (Дата)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_product ON sales_data (НоменклатураКод)')
    
    conn.commit()
    conn.close()

def save_to_database(df, db_path):
    """Сохранение данных в базу данных"""
    conn = sqlite3.connect(db_path)
    
    # Сохраняем данные
    df.to_sql('sales_data', conn, if_exists='append', index=False)
    
    conn.close()

def save_to_csv_by_shop(df, output_dir):
    """Сохранение данных в CSV файлы по магазинам"""
    
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем полный файл
    full_csv_path = os.path.join(output_dir, 'all_sales_data.csv')
    df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
    print(f"Полный файл сохранен: {full_csv_path} ({len(df)} строк)")
    
    # Сохраняем по магазинам
    for shop in df['Магазин'].unique():
        shop_data = df[df['Магазин'] == shop]
        shop_filename = f"{shop}.csv".replace(' ', '_').replace('/', '_')
        shop_path = os.path.join(output_dir, shop_filename)
        shop_data.to_csv(shop_path, index=False, encoding='utf-8-sig')
        print(f"Файл магазина сохранен: {shop_path} ({len(shop_data)} строк)")

def create_complete_time_series(df):
    """Создание полных временных рядов для анализа без добавления нулей"""
    
    # Получаем уникальные комбинации магазин-товар
    unique_combinations = df[['Магазин', 'НоменклатураКод', 'Страна', 'Категория', 'GAP']].drop_duplicates()
    
    # Получаем полный диапазон дат
    min_date = df['Дата'].min()
    max_date = df['Дата'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    print(f"Диапазон дат: {min_date} - {max_date} ({len(all_dates)} дней)")
    print(f"Уникальных комбинаций магазин-товар: {len(unique_combinations)}")
    
    return df

def main():
    """Основная функция обработки данных"""
    # Загрузка и предобработка данных
    sales_df, nomenclature_df = load_and_preprocess_data()
    
    # Схлопывание данных - ТОЛЬКО ФАКТИЧЕСКИЕ ПРОДАЖИ
    aggregated_sales = aggregate_sales_data(sales_df)
    
    # Слияние со справочником номенклатуры
    merged_sales = merge_with_nomenclature(aggregated_sales, nomenclature_df)
    
    # Определение GAP для оставшихся товаров
    final_sales = determine_gap_for_remaining(merged_sales, nomenclature_df)
    
    # Фильтрация данных
    # Удаляем категорию "Не алкоголь" и "Не напиток"
    final_sales = final_sales[final_sales['GAP'] != 'Не алкоголь']
    final_sales = final_sales[final_sales['GAP'] != 'Не напиток']

    # Исключаем магазин "Склад ТЗ Тау"
    final_sales = final_sales[final_sales['Магазин'] != 'Склад ТЗ Тау']

    # Применяем маппинг GAP к категориям
    final_sales['GAP'] = final_sales['GAP'].apply(map_gap_to_category)
    
    print(f"Данные после фильтрации: {len(final_sales)} строк")
    
    # УДАЛЯЕМ СТОЛБЕЦ НОМЕНКЛАТУРА ПЕРЕД СОХРАНЕНИЕМ
    final_sales = remove_nomenclature_column(final_sales)
    
    # Создаем полные временные ряды для анализа (без добавления нулей)
    analysis_data = create_complete_time_series(final_sales)
    
    # Настраиваем базу данных
    if os.path.exists(db_path):
        os.remove(db_path)
    setup_database(db_path)
    
    # Сохраняем в базу данных
    save_to_database(analysis_data, db_path)
    
    # Сохраняем в CSV
    save_to_csv_by_shop(analysis_data, output_dir)
    
    # Статистика
    print("\n=== СТАТИСТИКА ДАННЫХ ===")
    print(f"Всего записей: {len(analysis_data)}")
    print(f"Уникальных магазинов: {analysis_data['Магазин'].nunique()}")
    print(f"Уникальных товаров: {analysis_data['НоменклатураКод'].nunique()}")
    print(f"Диапазон дат: {analysis_data['Дата'].min()} - {analysis_data['Дата'].max()}")
    print(f"Уникальных дат: {analysis_data['Дата'].nunique()}")
    
    # Статистика по продажам
    print(f"\nСтатистика продаж:")
    print(f"Всего продаж: {analysis_data['Количество'].sum():.0f}")
    print(f"Среднее количество: {analysis_data['Количество'].mean():.2f}")
    print(f"Медиана количества: {analysis_data['Количество'].median():.2f}")
    print(f"Максимальная продажа: {analysis_data['Количество'].max():.0f}")
    
    # Статистика по магазинам
    print(f"\nСтатистика по магазинам:")
    for shop in analysis_data['Магазин'].unique():
        shop_data = analysis_data[analysis_data['Магазин'] == shop]
        print(f"  {shop}: {len(shop_data)} записей, {shop_data['Количество'].sum():.0f} продаж")
    
    print(f"\nОбработка завершена!")
    print(f"База данных: {db_path}")
    print(f"CSV файлы: {output_dir}")

if __name__ == "__main__":
    main()