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

# Параметры наличия товаров по категориям (в днях)
PRESENCE_PARAMETERS = {
    # Вино
    'Вино': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    'Столовое вино': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    'Плодово-фруктовое вино': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    'Креплёное вино': {'max_gap': 730, 'buffer': 180, 'min_period': 90},
    'Вермут': {'max_gap': 730, 'buffer': 180, 'min_period': 90},
    
    # Водка
    'Водка': {'max_gap': 730, 'buffer': 180, 'min_period': 90},
    
    # Ликер
    'Ликер': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    
    # Коньяк, бренди
    'Коньяк': {'max_gap': 1095, 'buffer': 270, 'min_period': 120},
    'Бренди': {'max_gap': 1095, 'buffer': 270, 'min_period': 120},
    
    # Виски, джин, ром
    'Виски': {'max_gap': 1095, 'buffer': 270, 'min_period': 120},
    'Джин': {'max_gap': 1095, 'buffer': 270, 'min_period': 120},
    'Ром': {'max_gap': 730, 'buffer': 180, 'min_period': 90},
    
    # Слабоалкогольные напитки
    'Пиво': {'max_gap': 180, 'buffer': 60, 'min_period': 30},
    'Сидр': {'max_gap': 180, 'buffer': 60, 'min_period': 30},
    'Медовуха': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    'Коктейль': {'max_gap': 365, 'buffer': 90, 'min_period': 60},
    
    # По умолчанию
    'default': {'max_gap': 365, 'buffer': 90, 'min_period': 60}
}

# Маппинг GAP из справочника к общим категориям
GAP_MAPPING = {
    "Вино/винные напитки": ["Вино", "Столовое вино", "Креплёное вино", "Вермут", "Газированное вино", "Плодово-фруктовое вино"],
    "Вино игристое": ["Вино игристое"],
    "ЛВИ": ["Виски", "Джин", "Ликер", "Ром", "Дистиллят", "Настойка", "Саке", "Аквавит"],
    "Слабоалкогольные напитки": ["Пиво", "Прочее"],
    "Водка": ["Водка"],
    "Бренди": ["Бренди"]
}

# Маппинг категорий алкоголя
ALCOHOL_CAT_MAPPING = {
    "Вино": "Вино",
    "Столовое вино": ["Вино безалкогольное", "Вино десертное", "Винный напиток"],
    "Креплёное вино": ["Вино креплёное", "Херес", "Мадера", "Марсала", "Портвейн", "Вино ликерное"],
    "Вермут": ["Вермут"],
    "Газированное вино": [],
    "Плодово-фруктовое вино": ["Вино фруктовое", "Плодовый напиток"],
    "Вино игристое": ["Шампанское", "Вино игристое"],
    "Пиво": ["Пиво"],
    "Прочее": ["Коктейль", "Медовуха", "Сидр"],
    "Виски": ["Бурбон", "Виски( Бурбон)", "Виски"],
    "Бренди": ["Коньяк", "Бренди", "Кальвадос", "Писко", "Чача", "Ракия", "Арманьяк"],
    "Джин": ["Джин"],
    "Аквавит": ["Аквавит"],
    "Ликер": ["Ликер"],
    "Ром": ["Ром", "Кашаса"],
    "Дистиллят": ["Дистиллят", "Шнапс", "Абсент"],
    "Настойка": ["Настойка", "Бальзам", "Висковый напиток", "Биттер"],
    "Саке": ["Саке"],
    "Водка": ["Водка", "Текила", "Граппа", "Соджу"]
}

# Дополнительные параметры на основе подкатегорий и видов
SUB_CATEGORY_PARAMS = {
    'бел': {'max_gap_multiplier': 1.0},
    'бел.': {'max_gap_multiplier': 1.0},
    'кр': {'max_gap_multiplier': 1.2},
    'крас': {'max_gap_multiplier': 1.2},
    'крас.': {'max_gap_multiplier': 1.2},
    'роз': {'max_gap_multiplier': 1.0}
}

PRODUCT_TYPE_PARAMS = {
    'бюрт': {'max_gap_multiplier': 0.8},
    'вино ликерное': {'max_gap_multiplier': 1.0},
    'п/слад.': {'max_gap_multiplier': 1.0},
    'п/сух.': {'max_gap_multiplier': 1.0},
    'сл': {'max_gap_multiplier': 1.0},
    'слад.': {'max_gap_multiplier': 1.0},
    'сух': {'max_gap_multiplier': 1.2},
    'сух.': {'max_gap_multiplier': 1.2}
}

def load_and_preprocess_data():
    """Загрузка и предварительная обработка данных"""
    
    # Загрузка данных о продажах
    sales_df = pd.read_excel(sales_path)
    print(f"Размер sales_df до обработки: {sales_df.shape}")
    
    # Удаляем ненужные столбцы
    columns_to_drop = ['Номер', 'ДисконтнаяКарта']
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
    """Схлопывание таблицы продаж"""
    print("Начинаем схлопывание таблицы...")
    
    # Группируем по ключевым полям
    grouped = df.groupby(['Дата', 'Магазин', 'Номенклатура', 'НоменклатураКод'])
    
    # Агрегируем данные
    aggregated = grouped.agg({
        'Количество': 'sum',
        'Сумма': 'sum',
        'Себестоимость': 'sum',
        'marginCost': 'sum',
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

def map_category_to_presence(category):
    """Маппинг категории к параметрам наличия"""
    # Если категория None, возвращаем категорию по умолчанию
    if category is None:
        return 'default'
    
    # Сначала проверяем точное совпадение
    if category in PRESENCE_PARAMETERS:
        return category
    
    # Затем проверяем маппинг алкогольных категорий
    for presence_category, alcohol_categories in ALCOHOL_CAT_MAPPING.items():
        if category in alcohol_categories or category == presence_category:
            if presence_category in PRESENCE_PARAMETERS:
                return presence_category
    
    # Если не нашли, используем категорию по умолчанию
    return 'default'

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
    
    # Добавляем недостающие столбцы в sales_data
    for col in ['Подкатегория', 'Вид']:
        if col not in sales_data.columns:
            sales_data[col] = None
    
    # Создаем словарь для быстрого доступа к данным справочника
    nom_dict = {}
    for _, row in nom_data.iterrows():
        key = (str(row['НоменклатураКод']), str(row['Номенклатура']))
        nom_dict[key] = {
            'GAP': row['GAP'],
            'Категория': row['Категория'],
            'Страна': row.get('Страна', None),
            'Подкатегория': row.get('Подкатегория', None),
            'Вид': row.get('Вид', None)
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
            for col in ['Страна', 'Подкатегория', 'Вид']:
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


def get_presence_parameters(category, subcategory, product_type):
    """Получение параметров наличия товара на основе категории, подкатегории и вида"""
    # Если категория None, используем категорию по умолчанию
    if category is None:
        category = 'default'
    
    # Определяем базовую категорию для параметров
    base_category = map_category_to_presence(category)
    
    # Получаем базовые параметры
    base_params = PRESENCE_PARAMETERS.get(base_category, PRESENCE_PARAMETERS['default']).copy()
    
    # Применяем модификаторы подкатегории (только если subcategory не None)
    if subcategory and pd.notna(subcategory):
        subcategory_mod = SUB_CATEGORY_PARAMS.get(subcategory, {}).get('max_gap_multiplier', 1.0)
        base_params['max_gap'] = int(base_params['max_gap'] * subcategory_mod)
    
    # Применяем модификаторы вида продукта (только если product_type не None)
    if product_type and pd.notna(product_type):
        product_type_mod = PRODUCT_TYPE_PARAMS.get(product_type, {}).get('max_gap_multiplier', 1.0)
        base_params['max_gap'] = int(base_params['max_gap'] * product_type_mod)
    
    return base_params

def determine_presence_periods(product_sales, category, subcategory, product_type):
    """Определение периодов наличия товара с учетом категориальных особенностей"""
    if len(product_sales) == 0:
        return []
    
    # Получаем параметры наличия для этой категории товара
    params = get_presence_parameters(category, subcategory, product_type)
    
    # Сортируем даты продаж
    dates = product_sales['Дата'].sort_values().unique()
    
    # Определяем периоды наличия
    periods = []
    current_start = dates[0]
    current_end = dates[0]
    
    for i in range(1, len(dates)):
        # Исправляем вычисление разницы в днях
        gap = (dates[i] - current_end).astype('timedelta64[D]').astype(int)
        
        if gap <= params['max_gap']:
            # Продолжаем текущий период
            current_end = dates[i]
        else:
            # Завершаем текущий период
            period_length = (current_end - current_start).astype('timedelta64[D]').astype(int)
            if period_length >= params['min_period']:
                # Добавляем буфер к концу периода
                period_end = current_end + np.timedelta64(params['buffer'], 'D')
                periods.append((current_start, period_end))
            
            # Начинаем новый период
            current_start = dates[i]
            current_end = dates[i]
    
    # Добавляем последний период
    period_length = (current_end - current_start).astype('timedelta64[D]').astype(int)
    if period_length >= params['min_period']:
        period_end = current_end + np.timedelta64(params['buffer'], 'D')
        periods.append((current_start, period_end))
    
    # Для товаров с одной продажей
    if len(dates) == 1:
        period_end = current_start + np.timedelta64(params['buffer'], 'D')
        periods = [(current_start, period_end)]
    
    return periods

def add_zero_sales_smart(data, all_dates_in_period, shop):
    """Добавление нулевых продаж только для периодов наличия товара"""
    # Если данных нет, возвращаем пустой DataFrame
    if len(data) == 0:
        return pd.DataFrame()
    
    # Получаем все уникальные товары
    unique_products_cols = ['Номенклатура', 'НоменклатураКод', 'Страна', 'Категория', 'GAP']
    if 'Подкатегория' in data.columns:
        unique_products_cols.append('Подкатегория')
    if 'Вид' in data.columns:
        unique_products_cols.append('Вид')
    
    unique_products = data[unique_products_cols].drop_duplicates()
    
    # Если нет уникальных товаров, возвращаем пустой DataFrame
    if len(unique_products) == 0:
        return pd.DataFrame()
    
    full_grid = []
    
    for _, product in unique_products.iterrows():
        product_code = product['НоменклатураКод']
        product_sales = data[data['НоменклатураКод'] == product_code]
        
        # Определяем периоды наличия товара
        periods = determine_presence_periods(
            product_sales, 
            product['Категория'], 
            product.get('Подкатегория', None), 
            product.get('Вид', None)
        )
        
        # Добавляем даты только в пределах периодов наличия
        for start_date, end_date in periods:
            period_dates = [d for d in all_dates_in_period if start_date <= d <= end_date]
            for date in period_dates:
                record = {
                    'Дата': date,
                    'Номенклатура': product['Номенклатура'],
                    'НоменклатураКод': product_code,
                    'Страна': product['Страна'],
                    'Категория': product['Категория'],
                    'GAP': product['GAP']
                }
                
                # Добавляем подкатегорию и вид, если они есть
                if 'Подкатегория' in product and pd.notna(product['Подкатегория']):
                    record['Подкатегория'] = product['Подкатегория']
                if 'Вид' in product and pd.notna(product['Вид']):
                    record['Вид'] = product['Вид']
                
                full_grid.append(record)
    
    # Если нет данных для добавления, возвращаем исходные данные
    if len(full_grid) == 0:
        return data
    
    # Создаем DataFrame из полной сетки
    full_grid_df = pd.DataFrame(full_grid)
    
    # Проверяем, есть ли общие столбцы для объединения
    common_columns = list(set(full_grid_df.columns) & set(data.columns))
    
    # Если нет общих столбцов, возвращаем полную сетку с нулевыми значениями
    if len(common_columns) == 0:
        result = full_grid_df.copy()
        # Добавляем нулевые значения для числовых столбцов
        numeric_cols = ['Количество', 'Сумма', 'Себестоимость', 'marginCost', 'promoFlag']
        for col in numeric_cols:
            if col in data.columns:
                result[col] = 0
    else:
        # Объединяем с исходными данными
        result = pd.merge(full_grid_df, data, 
                         on=common_columns, 
                         how='left')
    
    # Заполняем нулями отсутствующие продажи
    numeric_cols = ['Количество', 'Сумма', 'Себестоимость', 'marginCost', 'promoFlag']
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    
    # Добавляем недостающие столбцы
    result['Магазин'] = shop
    for col in ['is_holiday_day', 'Месяц', 'ГОД']:
        if col in result.columns:
            if col == 'Месяц':
                result[col] = result[col].fillna(result['Дата'].dt.month)
            elif col == 'ГОД':
                result[col] = result[col].fillna(result['Дата'].dt.year)
            else:
                result[col] = result[col].fillna(0)
    
    return result

def setup_database(db_path):
    """Настройка базы данных и создание таблиц"""
    conn = sqlite3.connect(db_path)
    
    # Создаем основную таблицу для хранения всех данных
    conn.execute('''
    CREATE TABLE IF NOT EXISTS sales_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Дата TEXT,
        Магазин TEXT,
        Номенклатура TEXT,
        НоменклатураКод TEXT,
        Страна TEXT,
        Категория TEXT,
        Подкатегория TEXT,
        Вид TEXT,
        GAP TEXT,
        Количество REAL,
        Сумма REAL,
        Себестоимость REAL,
        marginCost REAL,
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

def main():
    """Основная функция обработки данных"""
    # Загрузка и предобработка данных
    sales_df, nomenclature_df = load_and_preprocess_data()
    
    # Схлопывание данных
    aggregated_sales = aggregate_sales_data(sales_df)
    
    # Слияние со справочником номенклатуры
    merged_sales = merge_with_nomenclature(aggregated_sales, nomenclature_df)
    
    # Определение GAP для оставшихся товаров
    final_sales = determine_gap_for_remaining(merged_sales, nomenclature_df)
    
    # Удаляем категорию "Не алкоголь"
    final_sales = final_sales[final_sales['GAP'] != 'Не алкоголь']
    final_sales = final_sales[final_sales['GAP'] != 'Не напиток']

    # Исключаем магазин "Склад ТЗ Тау"
    final_sales = final_sales[final_sales['Магазин'] != 'Склад ТЗ Тау']

    # Применяем маппинг GAP к категориям
    final_sales['GAP'] = final_sales['GAP'].apply(map_gap_to_category)
    
    # Получаем уникальные магазины и даты
    unique_shops = final_sales['Магазин'].unique()
    all_dates = final_sales['Дата'].unique()
    
    print(f"Уникальные магазины: {len(unique_shops)}")
    print(f"Всего дат в периоде: {len(all_dates)}")
    print(f"Магазины для обработки: {list(unique_shops)}")
    
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Настраиваем базу данных
    if os.path.exists(db_path):
        os.remove(db_path)
    setup_database(db_path)
    
    # Обработка каждого магазина
    for shop in unique_shops:
        print(f"\nОбрабатываем магазин: {shop}")
        
        # Фильтруем данные по магазину
        shop_data = final_sales[final_sales['Магазин'] == shop].copy()
        
        # Получаем уникальные GAP для этого магазина
        unique_gaps = shop_data['GAP'].unique()
        print(f"  Найдено GAP: {unique_gaps}")
        
        # Обрабатываем каждый GAP
        for gap in unique_gaps:
            print(f"  Обрабатываем GAP: {gap}")
            
            # Фильтруем данные по GAP
            gap_data = shop_data[shop_data['GAP'] == gap].copy()
            
            if len(gap_data) == 0:
                print(f"    Пропускаем: нет данных")
                continue

            # Добавляем нулевые продажи с интеллектуальным определением периодов
            gap_data_full = add_zero_sales_smart(gap_data, all_dates, shop)
            
            # Сохраняем в базу данных
            save_to_database(gap_data_full, db_path)
            
            print(f"    Сохранено в БД: {len(gap_data_full)} строк")
    
    print()
    print("Обработка завершена!")
    print(f"Результаты сохранены в базе данных: {db_path}")
    
    # Создаем CSV файлы для удобства просмотра (только первые 1000 строк каждой категории)
    conn = sqlite3.connect(db_path)
    
    # Создаем папки для каждого магазина и сохраняем CSV файлы
    for shop in unique_shops:
        shop_dir = os.path.join(output_dir, shop)
        os.makedirs(shop_dir, exist_ok=True)
        
        # Получаем уникальные GAP для этого магазина из базы данных
        gaps_query = f"SELECT DISTINCT GAP FROM sales_data WHERE Магазин = '{shop}'"
        gaps = pd.read_sql_query(gaps_query, conn)['GAP'].tolist()
        
        for gap in gaps:
            # Читаем первые 1000 строк для этого магазина и GAP
            query = f"SELECT * FROM sales_data WHERE Магазин = '{shop}' AND GAP = '{gap}' LIMIT 1000"
            sample_data = pd.read_sql_query(query, conn)
            
            if len(sample_data) == 0:
                continue
            
            # Сохраняем в CSV
            csv_filename = f"{gap}.csv".replace(' ', '_').replace('.', '_').replace('/', '_').replace('\\', '_')
            csv_path = os.path.join(shop_dir, csv_filename)
            sample_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"    Создан CSV: {csv_path} ({len(sample_data)} строк)")
    
    conn.close()
    print("Готово!")

if __name__ == "__main__":
    main()
