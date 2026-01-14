"""
Модуль для обработки обезличивания данных
Упрощенная версия - пользователь выбирает только ГАПы и категории
"""

import pandas as pd
import sqlite3
import json
import os
from typing import Dict, List, Optional
from mapping_dict import *
from mapping_dict import ALCOHOL_CAT_MAPPING, REAL_CAT_TO_ANON_CAT
from item_cache import item_cache

class ItemAnonymizer:
    """Класс для обезличивания кодов товаров"""
    
    MAPPING_FILE = "item_mapping.json"
    
    def __init__(self):
        self.mapping = self._load_mapping()
        self.category_counters = self._init_category_counters()
    
    def _load_mapping(self) -> Dict:
        """Загружает существующий маппинг из файла"""
        if os.path.exists(self.MAPPING_FILE):
            try:
                with open(self.MAPPING_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_mapping(self):
        """Сохраняет маппинг в файл"""
        with open(self.MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
    
    def _init_category_counters(self) -> Dict:
        """Инициализирует счетчики для категорий"""
        counters = {}
        for real_code, anon_code in self.mapping.items():
            if '_' in anon_code:
                prefix = anon_code.split('_')[0]
                # Находим категорию по префиксу
                for category, cat_prefix in CATEGORY_PREFIXES.items():
                    if cat_prefix == prefix:
                        try:
                            num_part = anon_code.split('_')[1]
                            num = int(num_part)
                            if category not in counters or num > counters[category]:
                                counters[category] = num
                        except:
                            pass
                        break
        return counters
    
    def anonymize_item_code(self, real_code: str, category: str) -> str:
        """Создает обезличенный код для реального кода товара с учетом категории"""
        if real_code in self.mapping:
            return self.mapping[real_code]
        
        # Создаем новый код
        prefix = CATEGORY_PREFIXES.get(category, 'ITEM')
        
        # Увеличиваем счетчик для категории
        if category not in self.category_counters:
            self.category_counters[category] = 0
        self.category_counters[category] += 1
        
        # Формат: ПРЕФИКС_НОМЕР
        anonymized_code = f"{prefix}_{self.category_counters[category]:04d}"
        self.mapping[real_code] = anonymized_code
        self._save_mapping()
        
        return anonymized_code


class DataProcessor:
    """Основной класс для обработки данных"""
    
    def __init__(self, db_path: str = "only_aclo_final.db"):
        self.db_path = db_path
        self.item_anonymizer = ItemAnonymizer()
        self.alcohol_cat_mapping = ALCOHOL_CAT_MAPPING
        self.real_cat_to_anon_cat = REAL_CAT_TO_ANON_CAT
        item_cache.set_db_path(db_path)

    def check_database_categories(self):
        """Диагностическая функция: проверяет какие категории есть в БД"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT DISTINCT Категория FROM sales_data ORDER BY Категория"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print("=== КАТЕГОРИИ В БАЗЕ ДАННЫХ ===")
            print(f"Всего уникальных категорий: {len(df)}")
            
            # Группируем по типам
            wine_categories = [c for c in df['Категория'].dropna().unique() 
                            if 'вино' in str(c).lower() or 'шампан' in str(c).lower()]
            
            print("\nКатегории вина (все):")
            for cat in sorted(wine_categories):
                print(f"  - {cat}")
            
            # Проверяем маппинг
            print("\n=== ПРОВЕРКА МАППИНГА ===")
            for anon_cat, real_cats in self.alcohol_cat_mapping.items():
                found = []
                for real_cat in real_cats:
                    if real_cat in df['Категория'].values:
                        found.append(real_cat)
                
                if found:
                    print(f"{anon_cat}: найдено {len(found)} из {len(real_cats)} категорий")
                else:
                    print(f"{anon_cat}: не найдено ни одной категории из {real_cats}")
            
            return df['Категория'].tolist()
            
        except Exception as e:
            print(f"Ошибка при проверке категорий: {e}")
            return []
    
    def _get_items_with_sufficient_history_in_shops(
        self, 
        items: List[str], 
        shops: List[str],
        min_history_days: int = 14  # УМЕНЬШИТЕ С 30 ДО 14!
    ) -> List[str]:
        """
        Фильтрует товары по наличию достаточной истории В КОНКРЕТНЫХ МАГАЗИНАХ
        """
        if not items or not shops:
            print("Нет товаров или магазинов для проверки истории")
            return []
        
        # ОГРАНИЧИВАЕМ количество товаров для проверки
        if len(items) > 200:
            print(f"Ограничиваем проверку истории с {len(items)} до 200 товаров")
            items_to_check = items[:200]
        else:
            items_to_check = items
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Проверяем историю для каждого товара В КАЖДОМ МАГАЗИНЕ
            query = """
            SELECT 
                НоменклатураКод,
                Магазин,
                COUNT(DISTINCT Дата) as days_count
            FROM sales_data 
            WHERE НоменклатураКод IN ({}) AND Магазин IN ({})
            GROUP BY НоменклатураКод, Магазин
            """.format(
                ','.join(['?'] * len(items_to_check)),
                ','.join(['?'] * len(shops))
            )
            
            df = pd.read_sql_query(query, conn, params=items_to_check + shops)
            conn.close()
            
            # Находим товары, которые есть хотя бы в одном магазине с достаточной историей
            sufficient_items = df[df['days_count'] >= min_history_days]['НоменклатураКод'].unique().tolist()
            
            print(f"Проверка истории в магазинах:")
            print(f"  Проверено товаров: {len(items_to_check)}")
            print(f"  Проверено магазинов: {len(shops)}")
            print(f"  Товаров с достаточной историей: {len(sufficient_items)}")
            print(f"  Пропущено товаров: {len(items_to_check) - len(sufficient_items)}")
            
            # Для отладки: статистика по магазинам
            if not sufficient_items and len(items_to_check) > 0:
                print(f"  Детальная статистика по магазинам:")
                for shop in shops[:3]:  # Показываем только первые 3 магазина
                    shop_items = df[(df['Магазин'] == shop)]
                    if not shop_items.empty:
                        avg_days = shop_items['days_count'].mean()
                        max_days = shop_items['days_count'].max()
                        print(f"    {shop}: среднее {avg_days:.1f} дней, максимум {max_days} дней")
            
            return sufficient_items
            
        except Exception as e:
            print(f"Ошибка при проверке истории в магазинах: {e}")
            import traceback
            traceback.print_exc()
            # В случае ошибки возвращаем все товары
            return items_to_check
    
    def convert_user_selection_to_real(
        self,
        anon_shops: List[str],
        anon_gaps: List[str],
        anon_categories: List[str],
        scope: str = "specific"
    ) -> Dict:
        """
        Конвертирует выбор пользователя из обезличенного формата в реальный
        Сравнение БЕЗ УЧЕТА РЕГИСТРА для ГАПов!
        """
        result = {
            "real_shops": [],
            "real_gaps": [],
            "real_categories": [],
            "real_items": []
        }
        
        # 1. Конвертируем магазины
        if scope == "all":
            result["real_shops"] = list(REAL_TO_ANON_SHOPS.keys())
        else:
            for shop in anon_shops:
                if shop in ANON_TO_REAL_SHOPS:
                    result["real_shops"].append(ANON_TO_REAL_SHOPS[shop])
        
        # 2. Конвертируем ГАПы - регистронезависимое сравнение
        for gap in anon_gaps:
            if gap in ANON_TO_REAL_GAPS:
                result["real_gaps"].append(ANON_TO_REAL_GAPS[gap])
            else:
                # Пробуем найти ГАП без учета регистра
                gap_lower = gap.lower()
                for real_gap_key, real_gap_value in ANON_TO_REAL_GAPS.items():
                    if real_gap_key.lower() == gap_lower or real_gap_value.lower() == gap_lower:
                        result["real_gaps"].append(real_gap_value)
                        break
        
        # 3. Конвертируем категории
        for category in anon_categories:
            if category in ANON_TO_REAL_CATEGORIES:
                result["real_categories"].append(ANON_TO_REAL_CATEGORIES[category])
        
        # 4. Получаем товары
        if result["real_categories"]:
            result["real_items"] = self._get_items_from_db_by_categories(
                result["real_categories"],
                result["real_gaps"],
                result["real_shops"] if scope == "specific" else None
            )
        
        return result
    
    def _get_items_from_db_by_categories(
        self, 
        real_categories: List[str], 
        real_gaps: List[str],
        real_shops: List[str] = None
    ) -> List[str]:
        """
        Получает коды товаров из базы данных по категориям и ГАПам
        Сравнение БЕЗ УЧЕТА РЕГИСТРА!
        """
        if not real_categories:
            return []
        
        # РАСШИРЯЕМ КАТЕГОРИИ С ПОМОЩЬЮ ALCOHOL_CAT_MAPPING
        expanded_categories = []
        for category in real_categories:
            if category in self.alcohol_cat_mapping:
                expanded_categories.extend(self.alcohol_cat_mapping[category])
            else:
                expanded_categories.append(category)
        
        # Убираем дубликаты и преобразуем в нижний регистр для сравнения
        expanded_categories_lower = list(set([str(c).strip().lower() for c in expanded_categories if c]))
        
        if not expanded_categories_lower:
            print("Нет категорий для поиска после расширения")
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Сначала получаем ВСЕ товары с категориями
            base_query = "SELECT DISTINCT НоменклатураКод, Категория, GAP, Магазин FROM sales_data"
            
            if real_shops:
                # Для магазинов используем точное сравнение
                placeholders = ','.join(['?'] * len(real_shops))
                base_query += f" WHERE Магазин IN ({placeholders})"
                params = real_shops
            else:
                params = []
            
            df = pd.read_sql_query(base_query, conn, params=params)
            conn.close()
            
            if df.empty:
                print("Нет данных в БД")
                return []
            
            # ФИЛЬТРУЕМ В ПАМЯТИ - сравниваем категории без учета регистра
            df['Категория_нижний'] = df['Категория'].astype(str).str.lower().str.strip()
            
            # Фильтруем по категориям
            mask_cats = df['Категория_нижний'].isin(expanded_categories_lower)
            
            # Фильтруем по ГАПам если указаны
            if real_gaps:
                # Получаем реальные названия ГАПов
                real_gap_names = []
                for gap in real_gaps:
                    if gap in ANON_TO_REAL_GAPS:
                        real_gap_names.append(ANON_TO_REAL_GAPS[gap])
                
                if real_gap_names:
                    # Сравниваем ГАПы без учета регистра
                    df['GAP_нижний'] = df['GAP'].astype(str).str.lower().str.strip()
                    gap_lower = [str(g).lower().strip() for g in real_gap_names]
                    mask_gaps = df['GAP_нижний'].isin(gap_lower)
                    mask = mask_cats & mask_gaps
                else:
                    mask = mask_cats
            else:
                mask = mask_cats
            
            filtered_df = df[mask]
            
            if filtered_df.empty:
                print(f"Не найдено товаров для категорий: {real_categories}")
                print(f"   Расширенные категории (в нижнем регистре): {expanded_categories_lower}")
                if real_shops:
                    print(f"   в магазинах: {real_shops}")
                if real_gaps:
                    print(f"   с ГАПами: {real_gaps}")
                return []
            
            items = filtered_df["НоменклатураКод"].unique().tolist()
            print(f"Найдено {len(items)} товаров для категорий: {real_categories}")
            
            # Статистика
            if len(items) > 0:
                print(f"   Найдено категорий: {filtered_df['Категория'].nunique()}")
                print(f"   Примеры категорий: {filtered_df['Категория'].unique()[:5].tolist()}")
                print(f"   Найдено ГАПов: {filtered_df['GAP'].nunique()}")
                print(f"   Примеры ГАПов: {filtered_df['GAP'].unique()[:3].tolist()}")
            
            return items
            
        except Exception as e:
            print(f"Ошибка при получении товаров: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    # def _get_items_directly(
    #     self, 
    #     real_categories: List[str], 
    #     real_gaps: List[str]
    # ) -> List[str]:
    #     """
    #     Получает товары напрямую из таблицы sales_data
    #     (если нет таблицы product_categories)
    #     """
    #     try:
    #         conn = sqlite3.connect(self.db_path)
            
    #         # Получаем все уникальные товары
    #         query = "SELECT DISTINCT НоменклатураКод FROM sales_data"
    #         df = pd.read_sql_query(query, conn)
    #         conn.close()
            
    #         if df.empty:
    #             print("Нет товаров в таблице sales_data")
    #             return []
            
    #         # Временное решение: берем первые N товаров для тестирования
    #         items = df["НоменклатураКод"].head(50).tolist()
    #         print(f"Взято {len(items)} товаров для тестирования")
            
    #         return items
            
    #     except Exception as e:
    #         print(f"Критическая ошибка: {e}")
    #         return []
    
    def anonymize_forecast_result(
        self,
        df: pd.DataFrame,
        selected_anon_shops: List[str]
    ) -> pd.DataFrame:
        """
        ОПТИМИЗИРОВАННОЕ обезличивание DataFrame с результатами прогноза
        """
        import time
        if df.empty:
            return df
        
        df = df.copy()
        start_time = time.time()
        
        # 1. Обезличиваем магазины
        if "Магазин" in df.columns:
            df["Магазин"] = df["Магазин"].map(REAL_TO_ANON_SHOPS)
            
            if selected_anon_shops:
                df = df[df["Магазин"].isin(selected_anon_shops)]
        
        # 2. ОПТИМИЗИРОВАННОЕ обезличивание товаров
        if "НоменклатураКод" in df.columns:
            # Получаем ВСЕ категории одним вызовом
            unique_items = df["НоменклатураКод"].unique().tolist()
            item_categories = self._get_categories_for_items(unique_items)
            
            # Применяем маппинг через векторизованные операции
            df["Категория"] = df["НоменклатураКод"].map(item_categories)
            
            # Создаем обезличенные коды
            df["КодТовара"] = df.apply(
                lambda row: self.item_anonymizer.anonymize_item_code(
                    str(row["НоменклатураКод"]),
                    str(row["Категория"]) if pd.notna(row["Категория"]) else "Неизвестно"
                ),
                axis=1
            )
        
        # 3. Удаляем ненужные колонки
        columns_to_drop = [
            "Количество", "promoFlag", "НоменклатураКод", 
            "Категория", "GAP", "weekday", "time_idx", "is_holiday_day"
        ]
        
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 4. Переименовываем колонки
        column_mapping = {
            "Магазин": "Магазин",
            "Дата": "Дата",
            "pred_Количество": "ПрогнозПродаж"
        }
        
        df = df.rename(columns=column_mapping)
        
        # 5. Упорядочиваем колонки
        output_columns = ["Магазин", "КодТовара", "Дата", "ПрогнозПродаж"]
        df = df[[col for col in output_columns if col in df.columns]]
        
        # 6. Сортируем для удобства
        sort_columns = [col for col in ["Магазин", "КодТовара", "Дата"] if col in df.columns]
        if sort_columns:
            df = df.sort_values(sort_columns)
        
        elapsed_time = time.time() - start_time
        print(f"  Обезличивание выполнено за {elapsed_time:.2f} сек")
        
        return df.reset_index(drop=True)
    
    def _get_categories_for_items(self, item_codes: List[str]) -> Dict[str, str]:
        """
        Оптимизированное получение категорий для товаров
        Возвращает словарь: item_code -> category
        """
        if not item_codes:
            return {}
        
        result = {}
        uncached_items = []
        
        # 1. Проверяем кэш
        for item_code in item_codes:
            cached = item_cache.get_category(item_code)
            if cached:
                result[item_code] = cached
            else:
                uncached_items.append(item_code)
        
        # 2. Если есть несached товары, загружаем их из БД
        if uncached_items:
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Разбиваем на пакеты для больших списков
                batch_size = 1000
                for i in range(0, len(uncached_items), batch_size):
                    batch = uncached_items[i:i + batch_size]
                    
                    query = """
                    SELECT DISTINCT НоменклатураКод, Категория 
                    FROM sales_data 
                    WHERE НоменклатураКод IN ({})
                    """.format(','.join(['?'] * len(batch)))
                    
                    df = pd.read_sql_query(query, conn, params=batch)
                    
                    if not df.empty:
                        for _, row in df.iterrows():
                            item_code = row["НоменклатураКод"]
                            category = row["Категория"]
                            result[item_code] = category
                            # Обновляем кэш
                            item_cache.update_cache(item_code, category)
                
                conn.close()
                
                print(f"  Кэш обновлен: загружено {len(uncached_items)} товаров")
                
            except Exception as e:
                print(f"  Ошибка при загрузке категорий: {e}")
        
        return result
    def warm_up_cache(self):
        """Прогрев кэша категорий товаров"""
        print("Прогрев кэша категорий...")
        try:
            item_cache.warm_up_cache(self.db_path)
            stats = item_cache.get_stats()
            print(f"  Кэш готов: {stats['total_items']} товаров, {stats['total_categories']} категорий")
        except Exception as e:
            print(f"  Ошибка прогрева кэша: {e}")
        