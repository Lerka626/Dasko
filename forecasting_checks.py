# import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Для лучшей отладки CUDA ошибок
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Конфигурация аллокатора памяти
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
# import time
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import pandas as pd
# import numpy as np
# import torch
# import gc
# import logging
# import signal
# import sys
# from datetime import datetime
# import sqlite3
# from tqdm import tqdm 

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from pytorch_forecasting.data.encoders import NaNLabelEncoder
# from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, SMAPE

# warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
# warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

# from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# # ДОПОЛНИТЕЛЬНАЯ ОПТИМИЗАЦИЯ ПАМЯТИ
# def optimize_memory():
#     """Принудительная очистка памяти"""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

# # -----------------------------------------------------------------------------
# # НАСТРОЙКА ЛОГГИРОВАНИЯ И ОБРАБОТКИ ОШИБОК
# # -----------------------------------------------------------------------------
# def setup_logging():
#     """Настройка комплексного логирования"""
#     log_dir = os.path.join(BASE_DIR, "logs")
#     os.makedirs(log_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Основной логгер
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(log_dir, f"training_{timestamp}.log")),
#             logging.StreamHandler()
#         ]
#     )
    
#     # Логгер для ошибок
#     error_logger = logging.getLogger('error_logger')
#     error_logger.setLevel(logging.ERROR)
#     error_handler = logging.FileHandler(os.path.join(log_dir, f"errors_{timestamp}.log"))
#     error_logger.addHandler(error_handler)
    
#     return error_logger

# def signal_handler(sig, frame):
#     """Обработчик аварийного завершения"""
#     error_logger.error("ПРИНУДИТЕЛЬНОЕ ЗАВЕРШЕНИЕ ПОЛЬЗОВАТЕЛЕМ!")
#     torch.cuda.empty_cache()
#     sys.exit(1)

# # Регистрируем обработчики сигналов
# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

# # -----------------------------------------------------------------------------
# # МОНИТОРИНГ GPU И ТЕМПЕРАТУРЫ
# # -----------------------------------------------------------------------------
# # class GPUMonitor:
# #     """Мониторинг температуры и нагрузки GPU"""
# #     def __init__(self):
# #         self.temp_log = os.path.join(BASE_DIR, "logs", "gpu_temperature.csv")
# #         self.setup_temp_log()
        
# #     def setup_temp_log(self):
# #         """Создание файла для логирования температуры"""
# #         os.makedirs(os.path.dirname(self.temp_log), exist_ok=True)
# #         if not os.path.exists(self.temp_log):
# #             with open(self.temp_log, 'w') as f:
# #                 f.write("timestamp,epoch,step,temperature_C,gpu_utilization_%,memory_used_%\n")
    
# #     def get_gpu_stats(self):
# #         """Получение статистики GPU"""
# #         try:
# #             if USE_GPU:
# #                 # Температура
# #                 temperature = torch.cuda.temperature()
                
# #                 # Utilización
# #                 utilization = torch.cuda.utilization()
                
# #                 # Память
# #                 memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                
# #                 return temperature, utilization, memory_allocated
# #             return None, None, None
# #         except:
# #             return None, None, None
    
# #     def log_temperature(self, epoch, step):
# #         """Логирование температуры каждые 50 шагов"""
# #         if step % 50 == 0:
# #             temp, util, mem = self.get_gpu_stats()
# #             if temp is not None:
# #                 timestamp = datetime.now().isoformat()
# #                 with open(self.temp_log, 'a') as f:
# #                     f.write(f"{timestamp},{epoch},{step},{temp},{util},{mem:.1f}\n")
                
# #                 # Ограничение нагрузки GPU до 90%
# #                 if util is not None and util > 90:
# #                     logging.warning(f"GPU нагрузка {util}% > 80%! Делаю паузу...")
# #                     print(f"GPU нагрузка {util}% > 90%! Делаю паузу...")
# #                     time.sleep(2)
                
# #                 # Ограничение температуры
# #                 if temp > 80:
# #                     logging.warning(f"Температура GPU {temp}°C! Делаю паузу...")
# #                     print(f"Температура GPU {temp}°C! Делаю паузу...")
# #                     time.sleep(5)

# # -----------------------------------------------------------------------------
# # Быстрые настройки PyTorch
# # -----------------------------------------------------------------------------
# warnings.filterwarnings("ignore")
# torch.set_float32_matmul_precision("medium")

# # Ограничение памяти GPU
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.90)  # Максимум 90% памяти

# USE_GPU = torch.cuda.is_available()
# SUPPORTS_BF16 = USE_GPU and torch.cuda.is_bf16_supported()

# # Автоматическое определение precision для разных GPU
# if USE_GPU:
#     device_name = torch.cuda.get_device_name(0)
#     if "3090" in device_name:
#         print("RTX 3090 detected - using bfloat16 for best performance")
#         precision = "bf16"  # bfloat16 для RTX 3090
#     else:
#         precision = "32-true"  # по умолчанию
# else:
#     precision = "32-true"  # для CPU

# PIN_MEMORY = True if USE_GPU else False
# NUM_WORKERS = 4 if USE_GPU else 0

# # -----------------------------------------------------------------------------
# # КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ
# # -----------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, 'sales_data.db')  # Изменено с DATA_PATH
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Инициализация логирования
# error_logger = setup_logging()
# # gpu_monitor = GPUMonitor()

# logging.info(f"GPU available: {USE_GPU}")
# print(f"GPU available: {USE_GPU}")
# if USE_GPU:
#     logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
#     logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# print(f"GPU available: {USE_GPU}")

# # -----------------------------------------------------------------------------
# # ГИПЕРПАРАМЕТРЫ С НАКОПЛЕНИЕМ ГРАДИЕНТОВ
# # -----------------------------------------------------------------------------
# HORIZON_PARAMS = {
#     30: {
#         "learning_rate": 0.02,
#         "hidden_size": 32,
#         "lstm_layers": 1,
#         "attention_head_size": 1,
#         "dropout": 0.2,  # Увеличила для борьбы с переобучением
#         "hidden_continuous_size": 8,
#         "batch_size": 8,
#         "accumulate_grad_batches": 4,  # накопления 
#         "max_encoder_length": 45,
#         "max_epochs": 1,
#         "limit_train_batches": 0.1,  # 10% данных
#         "limit_val_batches": 1.0,   # всегда
#         "gradient_clip_val": 1.0,
#     },
#     90: {
#         "learning_rate": 0.01,
#         "hidden_size": 32,
#         "lstm_layers": 1,
#         "attention_head_size": 1,
#         "dropout": 0.3,  # Больше регуляризации
#         "hidden_continuous_size": 8,
#         "batch_size": 4,  # Уменьшенный батч для экономии памяти, можно понизить до 16, а нижний умножай на 2
#         "accumulate_grad_batches": 8,  # Накопление градиентов для 90 дней
#         "max_encoder_length": 20,  # ес че попробуй уменьшить до 30
#         "max_epochs": 1,
#         "limit_train_batches": 0.05,  # 10% данных для скорости
#         "limit_val_batches": 1.0,  # всегда
#         "gradient_clip_val": 0.3,
#     }
# }

# HORIZONS = [30, 90]

# # -----------------------------------------------------------------------------
# # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# # -----------------------------------------------------------------------------
# def clear_memory():
#     """Очистка памяти GPU и CPU"""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print("Память очищена")

# def print_memory_usage():
#     """Вывод информации об использовании памяти"""
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / 1024**3
#         reserved = torch.cuda.memory_reserved() / 1024**3
#         print(f"GPU память: {allocated:.1f}GB / {reserved:.1f}GB")
    
#     import psutil
#     memory = psutil.virtual_memory()
#     print(f"RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")

# # -----------------------------------------------------------------------------
# # НОВЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ
# # -----------------------------------------------------------------------------
# def load_data_from_db(db_path, shops=None, gaps=None, min_date=None, max_date=None):
#     """Загрузка данных из SQLite базы данных с оптимизацией памяти"""
#     print("Загрузка данных из базы данных с оптимизацией...")
    
#     conn = sqlite3.connect(db_path)
    
#     # Сначала получаем количество записей
#     count_query = "SELECT COUNT(*) FROM sales_data WHERE 1=1"
#     params = []
    
#     if shops:
#         placeholders = ','.join(['?' for _ in shops])
#         count_query += f" AND Магазин IN ({placeholders})"
#         params.extend(shops)
#     if gaps:
#         placeholders = ','.join(['?' for _ in gaps])
#         count_query += f" AND GAP IN ({placeholders})"
#         params.extend(gaps)
#     if min_date:
#         count_query += " AND Дата >= ?"
#         params.append(min_date)
#     if max_date:
#         count_query += " AND Дата <= ?"
#         params.append(max_date)
    
#     total_rows = pd.read_sql_query(count_query, conn, params=params).iloc[0,0]
#     print(f"Всего записей для загрузки: {total_rows}")
    
#     # Если данных слишком много - загружаем чанками
#     CHUNK_SIZE = 100000
#     if total_rows > 500000:
#         print(f"Много данных ({total_rows}), загружаю чанками по {CHUNK_SIZE}...")
#         chunks = []
        
#         base_query = """
#         SELECT Дата, Магазин, НоменклатураКод, Категория, GAP, Количество,
#             promoFlag, is_holiday_day
#         FROM sales_data WHERE 1=1
#         """
        
#         if shops:
#             base_query += f" AND Магазин IN ({','.join(['?' for _ in shops])})"
#         if gaps:
#             base_query += f" AND GAP IN ({','.join(['?' for _ in gaps])})"
#         if min_date:
#             base_query += " AND Дата >= ?"
#         if max_date:
#             base_query += " AND Дата <= ?"
        
#         base_query += " ORDER BY Магазин, НоменклатураКод, Дата"
        
#         for offset in range(0, total_rows, CHUNK_SIZE):
#             chunk_query = f"{base_query} LIMIT {CHUNK_SIZE} OFFSET {offset}"
#             chunk = pd.read_sql_query(chunk_query, conn, params=params)
#             chunks.append(chunk)
#             print(f"Загружено {min(offset + CHUNK_SIZE, total_rows)}/{total_rows} записей")
        
#         df = pd.concat(chunks, ignore_index=True)
#     else:
#         # Обычная загрузка для малых объемов
#         query = """
#         SELECT Дата, Магазин, НоменклатураКод, GAP, Количество, 
#                promoFlag, Месяц, ГОД, is_holiday_day
#         FROM sales_data WHERE 1=1
#         """
        
#         params = []
#         if shops:
#             placeholders = ','.join(['?' for _ in shops])
#             query += f" AND Магазин IN ({placeholders})"
#             params.extend(shops)
#         if gaps:
#             placeholders = ','.join(['?' for _ in gaps])
#             query += f" AND GAP IN ({placeholders})"
#             params.extend(gaps)
#         if min_date:
#             query += " AND Дата >= ?"
#             params.append(min_date)
#         if max_date:
#             query += " AND Дата <= ?"
#             params.append(max_date)
        
#         query += " ORDER BY Магазин, НоменклатураКод, Дата"
        
#         df = pd.read_sql_query(query, conn, params=params)
    
#     conn.close()
#     print(f"Загружено {len(df)} записей из базы данных")
#     return df

# def get_available_shops_and_gaps(db_path):
#     """Получение списка доступных магазинов и категорий из базы данных"""
#     conn = sqlite3.connect(db_path)
#     shops = pd.read_sql_query("SELECT DISTINCT Магазин FROM sales_data ORDER BY Магазин", conn)['Магазин'].tolist()
#     gaps = pd.read_sql_query("SELECT DISTINCT GAP FROM sales_data ORDER BY GAP", conn)['GAP'].tolist()
#     conn.close()
#     return shops, gaps


# # -----------------------------------------------------------------------------
# # Загрузка и подготовка (без изменений)
# # -----------------------------------------------------------------------------
# def detect_promo_anomalies(df, window=30, threshold=2.0):
#     """
#     Обнаружение аномалий в продажах, которые могут указывать на акции.
#     Использует скользящее среднее и стандартное отклонение.
#     """
#     df = df.sort_values(['Магазин', 'НоменклатураКод', 'Дата'])
    
#     # Для каждой группы вычисляем скользящее среднее и стандартное отклонение
#     df['rolling_mean'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
#         lambda x: x.rolling(window=window, min_periods=5).mean()
#     )
#     df['rolling_std'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
#         lambda x: x.rolling(window=window, min_periods=5).std()
#     )
    
#     # Заполняем пропуски
#     df['rolling_mean'] = df['rolling_mean'].fillna(method='bfill')
#     df['rolling_std'] = df['rolling_std'].fillna(df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform('std'))
#     df['rolling_std'] = df['rolling_std'].fillna(0)
    
#     # Вычисляем z-score (отклонение от среднего)
#     df['z_score'] = (df['Количество'] - df['rolling_mean']) / df['rolling_std'].replace(0, 1)
    
#     # Определяем аномалии (возможные акции)
#     df['is_promo_anomaly'] = (df['z_score'] > threshold).astype(int)
#         # Создаем признаки на основе аномалий
#     df['days_since_anomaly'] = df.groupby(['Магазин', 'НоменклатураКод'])['is_promo_anomaly'].transform(
#         lambda x: x.replace(1, np.nan).ffill().fillna(0).cumsum()
#     )
    
#     return df.drop(['rolling_mean', 'rolling_std', 'z_score'], axis=1)

# def create_seasonal_features(df):
#     """Создание сезонных и календарных фич"""
#     print("Создание сезонных фич...")
    
#     # Базовые календарные фичи (уже есть в вашем коде)
#     df['weekday'] = df['Дата'].dt.weekday.astype(str)
#     df['day'] = df['Дата'].dt.day.astype(str)
#     df['month'] = df['Дата'].dt.month.astype(str)
#     df['year'] = df['Дата'].dt.year.astype(str)
    
#     # Детальная недельная сезонность
#     df['is_friday'] = (df['Дата'].dt.weekday == 4).astype(int)
#     df['is_saturday'] = (df['Дата'].dt.weekday == 5).astype(int)
#     df['is_sunday'] = (df['Дата'].dt.weekday == 6).astype(int)
#     df['is_monday'] = (df['Дата'].dt.weekday == 0).astype(int)
    
#     # Финансовые циклы
#     df['is_end_of_month'] = (df['Дата'].dt.day >= 25).astype(int)
#     df['is_beginning_of_month'] = (df['Дата'].dt.day <= 7).astype(int)
#     df['is_salary_week'] = ((df['Дата'].dt.day >= 5) & (df['Дата'].dt.day <= 15)).astype(int)
    
#     print("Сезонные фичи созданы")
#     return df

# def create_smart_lag_features(df):
#     """Умные лаг-фичи, которые игнорируют нули"""
#     print("Создание умных лаг-фичей...")
#     df = df.sort_values(['Магазин', 'НоменклатураКод', 'Дата'])
    
#     # Создаем колонку с продажами только в дни реальных продаж
#     df['sales_non_zero'] = df['Количество'].replace(0, np.nan)
    
#     # Лаги считаем только по реальным продажам
#     for lag in [7, 14, 30]:
#         df[f'lag_{lag}'] = df.groupby(['Магазин', 'НоменклатураКод'])['sales_non_zero'].shift(lag)
#         df[f'lag_{lag}'] = df[f'lag_{lag}'].fillna(0)
#         print(f"Создан lag_{lag}")
    
#     # Скользящие средние (уже работают правильно с нулями)
#     for window in [7, 14, 30]:
#         df[f'rolling_mean_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
#             lambda x: x.rolling(window=window, min_periods=1).mean()
#         )
#         print(f"Создан rolling_mean_{window}")
    
#     df = df.drop('sales_non_zero', axis=1)
#     print("Умные лаг-фичи созданы")
#     return df


# def prepare_data_from_db(db_path, selected_shops=None, selected_gaps=None):
#     """Упрощенная и надежная версия подготовки данных"""
#     print("Загрузка данных из базы данных...")
    
#     # Загружаем данные из базы
#     df = load_data_from_db(db_path, shops=selected_shops, gaps=selected_gaps)
    
#     if len(df) == 0:
#         raise ValueError("Не удалось загрузить данные из базы данных")
    
#     # Удаляем проблемные столбцы (только если они существуют)
#     columns_to_drop = ['Вид', 'Подкатегория', 'Номенклатура', 'Страна']
#     existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
#     if existing_columns_to_drop:
#         df = df.drop(columns=existing_columns_to_drop)
#         print(f"Удалены столбцы: {existing_columns_to_drop}")
    
#     # Базовые преобразования
#     df['Дата'] = pd.to_datetime(df['Дата'])
#     df = df[df["Дата"].notnull() & df["Количество"].notnull()].copy()
    
#     # ДОБАВЛЯЕМ СОЗДАНИЕ ФИЧ
#     df = create_seasonal_features(df)
#     df = create_smart_lag_features(df)
    
#     # ПРЕОБРАЗОВАНИЕ В КАТЕГОРИИ - ИСПРАВЛЕННАЯ ВЕРСИЯ
#     categorical_columns = [
#         'Магазин', 'НоменклатураКод', 'Категория', 'GAP',
#         'promoFlag', 'is_holiday_day', 'weekday', 'day', 'month', 'year',
#         'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
#         'is_end_of_month', 'is_beginning_of_month', 'is_salary_week'
#     ]
    
#     for col in categorical_columns:
#         if col in df.columns:
#             # ПРЕОБРАЗУЕМ В СТРОКУ, ПОТОМ В КАТЕГОРИЮ
#             df[col] = df[col].astype(str)
#             df[col] = df[col].astype('category')
    
#     # ВРЕМЕННЫЕ ИНДЕКСЫ
#     min_date = df["Дата"].min()
#     df["time_idx"] = (df["Дата"] - min_date).dt.days
    
#     # ГОД и МЕСЯЦ как категории (не как числа)
#     df["ГОД"] = df["Дата"].dt.year.astype(str).astype('category')
#     df["Месяц"] = df["Дата"].dt.month.astype(str).astype('category')
    
#     print(f"Данные подготовлены: {len(df)} записей")
    
#     # Финальная проверка
#     print("Финальные типы данных:")
#     for col in ['weekday', 'day', 'promoFlag', 'month', 'year']:
#         if col in df.columns:
#             print(f"  {col}: {df[col].dtype}")
    
#     return df

# def aggregate_daily_data(df):
#     """
#     Агрегирует данные по дням для каждой комбинации магазин-товар
#     """
#     print("Агрегация данных по дням...")
    
#     # Проверяем реальные NaN в датах
#     nan_dates_before = df['Дата'].isnull().sum()
#     print(f"Реальные NaN в датах до агрегации: {nan_dates_before}")
    
#     if nan_dates_before > 0:
#         df = df.dropna(subset=['Дата'])
    
#     available_columns = df.columns.tolist()
    
#     # Базовые колонки
#     base_columns = ['Магазин', 'НоменклатураКод', 'time_idx', 'Дата', 'Количество']
    
#     # Дополнительные колонки - ДОБАВИЛИ НОВЫЕ ФИЧИ
#     optional_columns = ['promoFlag', 'is_holiday_day', 'Месяц', 'ГОД', 'weekday', 'day',
#                        'month', 'year', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
#                        'is_end_of_month', 'is_beginning_of_month', 'is_salary_week',
#                        'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
#                        'GAP', 'Категория']  # Добавили Категория
    
#     # Формируем финальный список
#     keep_columns = base_columns.copy()
#     for col in optional_columns:
#         if col in available_columns:
#             keep_columns.append(col)
    
#     print(f"Сохраняем столбцы: {keep_columns}")
#     df = df[keep_columns].copy()
    
#     # Определяем колонки для агрегации
#     numeric_cols = [col for col in ['Количество', 'lag_7', 'lag_14', 'lag_30', 
#                                    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30'] if col in df.columns]
#     categorical_cols = [col for col in ['promoFlag', 'is_holiday_day', 'Месяц', 'ГОД', 
#                                        'GAP', 'Категория', 'month', 'year',
#                                        'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
#                                        'is_end_of_month', 'is_beginning_of_month', 'is_salary_week'] if col in df.columns]
    
#     # Создаем словарь для агрегации
#     agg_dict = {}
#     for col in numeric_cols:
#         agg_dict[col] = 'sum' if col == 'Количество' else 'mean'  # Лаги и средние берем среднее
#     for col in categorical_cols:
#         agg_dict[col] = 'first'  # Берем первое значение
    
#     # ВАЖНОЕ ИСПРАВЛЕНИЕ: группируем и сохраняем дату
#     group_columns = ['Магазин', 'НоменклатураКод', 'time_idx']
    
#     # Сначала получим даты для каждой группы
#     date_mapping = df.groupby(group_columns)['Дата'].first().reset_index()
    
#     # Затем агрегируем остальные данные
#     df_agg = df.groupby(group_columns).agg(agg_dict).reset_index()
    
#     # Объединяем с датами
#     df_agg = df_agg.merge(date_mapping, on=group_columns, how='left')
    
#     # Проверяем даты после агрегации
#     nan_dates_after = df_agg['Дата'].isnull().sum()
#     print(f"NaN в датах после агрегации: {nan_dates_after}")
    
#     if nan_dates_after > 0:
#         print(f"Удаляем {nan_dates_after} строк с NaN в дате")
#         df_agg = df_agg.dropna(subset=['Дата'])
    
#     # Восстанавливаем календарные фичи
#     df_agg["weekday"] = df_agg['Дата'].dt.weekday.astype(str)
#     df_agg["day"] = df_agg['Дата'].dt.day.astype(str)
#     df_agg["month"] = df_agg['Дата'].dt.month.astype(str)
#     df_agg["year"] = df_agg['Дата'].dt.year.astype(str)
    
#     # Преобразуем категориальные колонки
#     categorical_columns = ['promoFlag', 'is_holiday_day', 'Месяц', 'ГОД', 'GAP', 'Категория', 
#                           'weekday', 'day', 'month', 'year',
#                           'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
#                           'is_end_of_month', 'is_beginning_of_month', 'is_salary_week']
#     for col in categorical_columns:
#         if col in df_agg.columns:
#             if df_agg[col].dtype != 'object':
#                 df_agg[col] = df_agg[col].astype(str)
#             df_agg[col] = df_agg[col].astype('category')
    
#     print(f"После агрегации: {len(df_agg)} записей")
    
#     return df_agg

# # def analyze_time_points(df):
# #     """
# #     Анализ количества дней с продажами для каждого товара в каждом магазине
# #     """
# #     # Группируем по магазину и товару
# #     analysis = df.groupby(['Магазин', 'НоменклатураКод']).agg({
# #         'Дата': ['count', 'min', 'max'],
# #         'Количество': 'sum'
# #     }).reset_index()
    
# #     # Упрощаем имена столбцов
# #     analysis.columns = ['Магазин', 'НоменклатураКод', 'days_with_sales', 'first_sale_date', 'last_sale_date', 'total_quantity']
    
# #     # Вычисляем длительность периода продаж в днях
# #     analysis['sales_period_days'] = (analysis['last_sale_date'] - analysis['first_sale_date']).dt.days + 1
    
# #     # Вычисляем заполненность периода продажами
# #     analysis['sales_coverage'] = analysis['days_with_sales'] / analysis['sales_period_days']
    
# #     # Сохраняем анализ в файл
# #     analysis_path = os.path.join(BASE_DIR, "sales_days_analysis.csv")
# #     analysis.to_csv(analysis_path, index=False)
    
# #     # Выводим статистику
# #     print("=== АНАЛИЗ ДНЕЙ С ПРОДАЖАМИ ===")
# #     print(f"Всего уникальных комбинаций магазин-товар: {len(analysis)}")
# #     print(f"Среднее количество дней с продажами на товар: {analysis['days_with_sales'].mean():.1f}")
# #     print(f"Медианное количество дней с продажами на товар: {analysis['days_with_sales'].median():.1f}")
# #     print(f"Минимальное количество дней с продажами: {analysis['days_with_sales'].min()}")
# #     print(f"Максимальное количество дней с продажами: {analysis['days_with_sales'].max()}")
    
# #     # Анализ заполненности периодов продажами
# #     coverage_stats = analysis['sales_coverage'].describe()
# #     print(f"Заполненность периодов продажами:")
# #     print(f"  Средняя: {coverage_stats['mean']:.2%}")
# #     print(f"  Медианная: {analysis['sales_coverage'].median():.2%}")
    
# #     # Анализ товаров с малым количеством дней с продажами
# #     low_sales_products = analysis[analysis['days_with_sales'] < 10]
# #     if len(low_sales_products) > 0:
# #         print(f"Найдено {len(low_sales_products)} товаров с малым количеством дней с продажами (<10):")
# #         # Группируем по магазинам для удобства чтения
# #         store_stats = low_sales_products.groupby('Магазин').size()
# #         for store, count in store_stats.items():
# #             print(f"  - {store}: {count} товаров")
    
# #     return analysis

# def split_by_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.2):
#     """Разделение для каждой группы отдельно с правильной логикой"""
#     results = []
    
#     # Собираем все уникальные группы (магазины)
#     all_groups = df["Магазин"].unique()
#     print(f"Всего уникальных магазинов: {len(all_groups)}")
    
#     # Анализируем распределение длин рядов
#     group_lengths = []
#     for group_name in all_groups:
#         group_data = df[df["Магазин"] == group_name]
#         length = len(group_data)
#         group_lengths.append(length)
#         print(f"Группа {group_name}: {length} записей")
    
#     print(f"Статистика длин рядов по группам:")
#     print(f"  Минимум: {min(group_lengths)} записей")
#     print(f"  Максимум: {max(group_lengths)} записей") 
#     print(f"  Медиана: {np.median(group_lengths):.1f} записей")
#     print(f"  Среднее: {np.mean(group_lengths):.1f} записей")
    
#     groups_used = 0
#     groups_with_issues = 0

#     categorical_columns = ['promoFlag', 'is_holiday_day', 'weekday', 'day', 'Категория', 'GAP', 'Страна']
#     # Проверяем, что ГОД и Месяц действительно числовые
#     # if 'ГОД' in df.columns:
#     #     if df['ГОД'].dtype.name == 'category':
#     #         print("Предупреждение: ГОД является категорией, преобразую в число")
#     #         df['ГОД'] = df['ГОД'].astype(str).str.extract('(\d+)').astype(float)
    
#     # if 'Месяц' in df.columns:
#     #     if df['Месяц'].dtype.name == 'category':
#     #         print("Предупреждение: Месяц является категорией, преобразую в число")
#     #         df['Месяц'] = df['Месяц'].astype(str).str.extract('(\d+)').astype(float)

#     all_categories = {}
#     for col in categorical_columns:
#         if col in df.columns:
#             all_categories[col] = set(df[col].unique())
#             print(f"Все категории для {col}: {list(all_categories[col])[:5]}...")  # Показываем первые 5
    
#     for group_name in all_groups:
#         group_data = df[df["Магазин"] == group_name].copy()
        
#         # ИСПРАВЛЕНИЕ: проверяем, что в группе достаточно данных
#         if len(group_data) < 5:
#             print(f"Группа {group_name} имеет только {len(group_data)} записей - добавляем в train")
#             df_train_group = group_data.copy()
#             df_val_group = pd.DataFrame()
#             df_test_group = pd.DataFrame()
#             groups_with_issues += 1
#         else:
#             # Сортируем по времени
#             group_data = group_data.sort_values("time_idx")
            
#             # ИСПРАВЛЕНИЕ: используем относительное разделение внутри группы
#             time_indices = sorted(group_data["time_idx"].unique())
#             n_times = len(time_indices)
            
#             train_cutoff_idx = int(train_ratio * n_times)
#             val_cutoff_idx = int((train_ratio + val_ratio) * n_times)
            
#             train_times = time_indices[:train_cutoff_idx]
#             val_times = time_indices[train_cutoff_idx:val_cutoff_idx] 
#             test_times = time_indices[val_cutoff_idx:]
            
#             df_train_group = group_data[group_data["time_idx"].isin(train_times)].copy()
#             df_val_group = group_data[group_data["time_idx"].isin(val_times)].copy()
#             df_test_group = group_data[group_data["time_idx"].isin(test_times)].copy()
            
#             # ИСПРАВЛЕНИЕ: если в val/test нет данных, берем последние данные из train
#             if len(df_val_group) == 0 and len(df_train_group) > 10:
#                 # Берем последние 20% train данных для val
#                 val_size = max(1, int(0.2 * len(df_train_group)))
#                 df_val_group = df_train_group.tail(val_size).copy()
#                 df_train_group = df_train_group.iloc[:-val_size].copy()
            
#             if len(df_test_group) == 0 and len(df_train_group) > 10:
#                 # Берем последние 10% train данных для test  
#                 test_size = max(1, int(0.1 * len(df_train_group)))
#                 df_test_group = df_train_group.tail(test_size).copy()
#                 df_train_group = df_train_group.iloc[:-test_size].copy()
        
#         # Проверяем, что в train есть данные
#         if len(df_train_group) == 0:
#             print(f"В группе {group_name} нет данных в тренировочном наборе")
#             # Используем все данные для train
#             df_train_group = group_data.copy()
#             df_val_group = pd.DataFrame()
#             df_test_group = pd.DataFrame()
#             groups_with_issues += 1
            
#         results.append((df_train_group, df_val_group, df_test_group))
#         groups_used += 1
        
#         if groups_used <= 10:  # Показываем только первые 10 групп для экономии места
#             print(f"  {group_name}: train={len(df_train_group)}, val={len(df_val_group)}, test={len(df_test_group)}")
    
#     if not results:
#         raise ValueError("Нет данных для обучения после разделения!")
    
#     # Объединяем результаты
#     df_train = pd.concat([r[0] for r in results], ignore_index=True)
#     df_val = pd.concat([r[1] for r in results], ignore_index=True)
#     df_test = pd.concat([r[2] for r in results], ignore_index=True)
    
#     print(f"\nИтоговое разделение:")
#     print(f"  Train: {len(df_train)} записей, {df_train['Магазин'].nunique()} групп")
#     print(f"  Val: {len(df_val)} записей, {df_val['Магазин'].nunique() if len(df_val) > 0 else 0} групп")
#     print(f"  Test: {len(df_test)} записей, {df_test['Магазин'].nunique() if len(df_test) > 0 else 0} групп")
#     print(f"  Групп с проблемами: {groups_with_issues}")
    
#     # Сохраняем категории из train для val и test
#     for col in df_train.columns:
#         if col in ['ГОД', 'Месяц', 'time_idx', 'Количество']:
#             continue
#         if df_train[col].dtype.name == 'category':
#             all_unique_categories = set()
#             for df_part in [df_train, df_val, df_test]:
#                 if len(df_part) > 0 and col in df_part.columns:
#                     all_unique_categories.update(df_part[col].unique())
            
#             # Создаем общий список категорий
#             common_categories = sorted(list(all_unique_categories))
            
#             # Применяем общие категории ко всем наборам
#             df_train[col] = df_train[col].astype(str).astype('category').cat.set_categories(common_categories)
#             if len(df_val) > 0:
#                 df_val[col] = df_val[col].astype(str).astype('category').cat.set_categories(common_categories)
#             if len(df_test) > 0:
#                 df_test[col] = df_test[col].astype(str).astype('category').cat.set_categories(common_categories)
            
#             print(f"Унифицированы категории для {col}: {len(common_categories)} уникальных значений")
#     # for col in ['ГОД', 'Месяц']:
#     #     if col in df_train.columns:
#     #         for df_part in [df_train, df_val, df_test]:
#     #             if len(df_part) > 0 and col in df_part.columns:
#     #                 if df_part[col].dtype.name == 'category':
#     #                     print(f"Преобразую {col} в число в {df_part.__class__.__name__}")
#     #                     df_part[col] = df_part[col].astype(str).str.extract('(\d+)').astype(float)
#     return df_train, df_val, df_test

# def check_shop_data_distribution(df):
#     """Проверка распределения данных по магазинам"""
#     print("\n=== РАСПРЕДЕЛЕНИЕ ДАННЫХ ПО МАГАЗИНАМ ===")
    
#     shop_stats = df.groupby('Магазин').agg({
#         'НоменклатураКод': 'nunique',
#         'Дата': ['min', 'max', 'nunique'],
#         'time_idx': ['min', 'max']
#     }).reset_index()
    
#     shop_stats.columns = ['Магазин', 'уникальных_товаров', 'первая_дата', 'последняя_дата', 'уникальных_дат', 'min_time_idx', 'max_time_idx']
    
#     shop_stats['дней_данных'] = (shop_stats['последняя_дата'] - shop_stats['первая_дата']).dt.days + 1
#     shop_stats['записей'] = df.groupby('Магазин').size().values
    
#     print(shop_stats.to_string(index=False))
    
#     return shop_stats

# def scale_numeric(train_df, val_df, test_df):
#     """Упрощенное масштабирование"""
#     # Только целевая переменная и лаг-фичи
#     num_cols = ["Количество", "lag_7", "lag_14", "lag_30", 
#                 "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]
    
#     # Оставляем только существующие колонки
#     num_cols = [col for col in num_cols if col in train_df.columns]
    
#     print(f"Масштабируемые столбцы: {num_cols}")
    
#     if not num_cols:
#         print("Предупреждение: нет численных столбцов для масштабирования")
#         return train_df, val_df, test_df, None
    
#     scaler = StandardScaler()
#     train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    
#     if len(val_df) > 0:
#         val_df[num_cols] = scaler.transform(val_df[num_cols])
    
#     if len(test_df) > 0:
#         test_df[num_cols] = scaler.transform(test_df[num_cols])
        
#     return train_df, val_df, test_df, scaler

# # -----------------------------------------------------------------------------
# # Dataset / Dataloaders (без изменений)
# # -----------------------------------------------------------------------------
# def make_dataset(df: pd.DataFrame, prediction_length: int, max_encoder_length: int = 60) -> TimeSeriesDataSet:
#     print("Создание датасета...")
#     print(f"Размер данных: {len(df)}")
    
#     if len(df) == 0:
#         raise ValueError("Передан пустой DataFrame")
    
#     # Проверяем наличие обязательных колонок
#     required_columns = ["time_idx", "Количество", "Магазин"]
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Отсутствует обязательная колонка: {col}")
    
#     # Убеждаемся, что все категориальные колонки имеют правильный тип
#     categorical_cols = ["promoFlag", "is_holiday_day", "weekday", "day", "Магазин"]
#     for col in categorical_cols:
#         if col in df.columns and df[col].dtype.name != 'category':
#             print(f"Преобразуем {col} в категорию")
#             # Сначала в строку, затем в категорию для безопасности
#             df[col] = df[col].astype(str).astype('category')
    
#     try:
#         # Упрощенная конфигурация с минимальными параметрами
#         dataset = TimeSeriesDataSet(
#             df,
#             time_idx="time_idx",
#             target="Количество",
#             group_ids=["Магазин"],
#             min_encoder_length=1,
#             max_encoder_length=max_encoder_length,
#             min_prediction_length=prediction_length,
#             max_prediction_length=prediction_length,
#             static_categoricals=["Магазин"],
#             time_varying_known_categoricals=["promoFlag", "is_holiday_day", "weekday", "day"],
#             time_varying_known_reals=["time_idx"],
#             time_varying_unknown_reals=["Количество"],
#             add_relative_time_idx=True,
#             add_target_scales=True,
#             add_encoder_length=True,
#             allow_missing_timesteps=True,
#         )
#         return dataset
#     except Exception as e:
#         print(f"Ошибка создания датасета: {e}")
#         # Пробуем упрощенную версию
#         try:
#             print("Пробуем упрощенную конфигурацию датасета...")
#             dataset = TimeSeriesDataSet(
#                 df,
#                 time_idx="time_idx",
#                 target="Количество",
#                 group_ids=["Магазин"],
#                 min_encoder_length=1,
#                 max_encoder_length=max_encoder_length,
#                 min_prediction_length=prediction_length,
#                 max_prediction_length=prediction_length,
#                 static_categoricals=["Магазин"],
#                 time_varying_known_reals=["time_idx"],
#                 time_varying_unknown_reals=["Количество"],
#                 add_relative_time_idx=True,
#                 add_target_scales=True,
#                 add_encoder_length=True,
#                 allow_missing_timesteps=True,
#             )
#             return dataset
#         except Exception as e2:
#             print(f"Ошибка в упрощенной конфигурации: {e2}")
#             raise

# def make_dataloader(ts_ds: TimeSeriesDataSet, train: bool, batch_size: int):
#     return ts_ds.to_dataloader(
#         train=train,
#         batch_size=batch_size,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         persistent_workers=(NUM_WORKERS > 0),
#         drop_last=False,  #train,
#         shuffle=train,  # Добавьте shuffle для обучения
#     )

# # class FixedTFT(TemporalFusionTransformer):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
    
# #     def training_step(self, batch, batch_idx):
# #         try:
# #             # УПРОЩЕННАЯ ВЕРСИЯ: передаем батч как есть
# #             output = self(batch)
            
# #             # Пробуем разные способы получения loss
# #             if hasattr(output, 'loss'):
# #                 loss = output.loss
# #             elif isinstance(output, dict) and 'loss' in output:
# #                 loss = output['loss']
# #             elif isinstance(output, (list, tuple)) and len(output) > 0:
# #                 # Если output - это кортеж/список, берем первый элемент
# #                 first_output = output[0]
# #                 if hasattr(first_output, 'loss'):
# #                     loss = first_output.loss
# #                 elif isinstance(first_output, dict) and 'loss' in first_output:
# #                     loss = first_output['loss']
# #                 else:
# #                     # Пробуем вычислить loss вручную
# #                     loss = self.compute_loss(output, batch)
# #             else:
# #                 # Последняя попытка - вычисляем loss вручную
# #                 loss = self.compute_loss(output, batch)
            
# #             if torch.isnan(loss).any() or torch.isinf(loss).any():
# #                 print(f"Обнаружены NaN/Inf в loss: {loss}")
# #                 return None
                
# #             self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
# #             return loss
            
# #         except Exception as e:
# #             if batch_idx == 0:
# #                 print(f"Первая ошибка в training_step: {e}")
# #                 print(f"Тип батча: {type(batch)}")
# #                 print(f"Тип output: {type(output) if 'output' in locals() else 'N/A'}")
# #                 if hasattr(output, '__dict__'):
# #                     print(f"Атрибуты output: {output.__dict__.keys()}")
# #             return None
    
# #     def validation_step(self, batch, batch_idx):
# #         try:
# #             output = self(batch)
            
# #             # Такая же логика получения loss как в training_step
# #             if hasattr(output, 'loss'):
# #                 loss = output.loss
# #             elif isinstance(output, dict) and 'loss' in output:
# #                 loss = output['loss']
# #             elif isinstance(output, (list, tuple)) and len(output) > 0:
# #                 first_output = output[0]
# #                 if hasattr(first_output, 'loss'):
# #                     loss = first_output.loss
# #                 elif isinstance(first_output, dict) and 'loss' in first_output:
# #                     loss = first_output['loss']
# #                 else:
# #                     loss = self.compute_loss(output, batch)
# #             else:
# #                 loss = self.compute_loss(output, batch)
                
# #             self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
# #             return loss
# #         except Exception as e:
# #             print(f"Ошибка в validation_step: {e}")
# #             return None
    
# #     def compute_loss(self, output, batch):
# #         """Ручное вычисление loss если автоматическое не работает"""
# #         try:
# #             # Получаем prediction и target
# #             if hasattr(output, 'prediction'):
# #                 prediction = output.prediction
# #             elif isinstance(output, dict) and 'prediction' in output:
# #                 prediction = output['prediction']
# #             else:
# #                 prediction = output[0] if isinstance(output, (list, tuple)) else output
            
# #             # Получаем target из batch
# #             if isinstance(batch, (list, tuple)) and len(batch) == 2:
# #                 # batch = (x, y) где y = [decoder_target, target]
# #                 targets = batch[1][0]  # decoder_target
# #             elif isinstance(batch, dict):
# #                 targets = batch.get('decoder_target', batch.get('target'))
# #             else:
# #                 targets = batch[1] if len(batch) > 1 else None
            
# #             if prediction is None or targets is None:
# #                 return torch.tensor(0.0, requires_grad=True)
            
# #             # Вычисляем SMAPE вручную
# #             return self.loss(prediction, targets)
            
# #         except Exception as e:
# #             print(f"Ошибка в compute_loss: {e}")
# #             return torch.tensor(0.0, requires_grad=True)

# #     def predict_step(self, batch, batch_idx, dataloader_idx=0):
# #         try:
# #             output = self(batch)
# #             if hasattr(output, 'prediction'):
# #                 return output.prediction
# #             elif isinstance(output, dict) and 'prediction' in output:
# #                 return output['prediction']
# #             else:
# #                 return output
# #         except Exception as e:
# #             print(f"Ошибка в predict_step: {e}")
# #             return None
        

# def analyze_group_timeseries(df, group_col="Магазин", time_col="time_idx"):
#     """Детальный анализ временных рядов для каждой группы"""
#     analysis = df.groupby(group_col).agg({
#         time_col: ['min', 'max', 'count', 'nunique'],
#         'Дата': ['min', 'max']
#     }).reset_index()
    
#     analysis.columns = [group_col, 'min_time', 'max_time', 'data_points', 'unique_times', 'first_date', 'last_date']
#     analysis['time_range'] = analysis['max_time'] - analysis['min_time'] + 1
#     analysis['coverage'] = analysis['data_points'] / analysis['time_range']
#     analysis['missing_days'] = analysis['time_range'] - analysis['data_points']
    
#     print("=== АНАЛИЗ ВРЕМЕННЫХ РЯДОВ ПО ГРУППАМ ===")
#     print(analysis.to_string(index=False))
    
#     return analysis

# def create_adaptive_dataset(df, prediction_length, max_encoder_length=60):
#     """Создает датасет с адаптивной длиной энкодера и защитой от фильтрации"""
#     print("Создание адаптивного датасета (сохраняем все данные)...")
#     print(f"Размер данных: {len(df)}")
    
#     if len(df) == 0:
#         raise ValueError("Передан пустой DataFrame")
    
#     # Анализируем длину рядов
#     group_stats = df.groupby('Магазин')['time_idx'].agg(['min', 'max', 'count']).reset_index()
#     group_stats['available_length'] = group_stats['max'] - group_stats['min'] + 1
    
#     print("Статистика по группам:")
#     print(f"Всего групп: {len(group_stats)}")
#     print(f"Средняя длина ряда: {group_stats['available_length'].mean():.1f} дней")
#     print(f"Минимальная длина: {group_stats['available_length'].min()} дней")
#     print(f"Максимальная длина: {group_stats['available_length'].max()} дней")
    
#     # АДАПТИВНАЯ ДЛИНА ЭНКОДЕРА с защитой от коротких рядов
#     min_available_length = group_stats['available_length'].min()
    
#     # ВАЖНОЕ ИСПРАВЛЕНИЕ: учитываем prediction_length при расчете
#     required_total_length = max_encoder_length + prediction_length
    
#     if min_available_length < required_total_length:
#         print(f"ВНИМАНИЕ: минимальная длина ряда ({min_available_length}) меньше требуемой ({required_total_length})")
#         # Адаптивно уменьшаем encoder_length
#         adaptive_encoder_length = max(5, min_available_length - prediction_length - 5)  # Оставляем запас
#         print(f"Адаптивная длина энкодера уменьшена до: {adaptive_encoder_length} дней")
#     else:
#         adaptive_encoder_length = min(max_encoder_length, int(min_available_length * 0.7))
    
#     adaptive_encoder_length = max(5, adaptive_encoder_length)  # Минимум 5 дней
    
#     print(f"Используемая длина энкодера: {adaptive_encoder_length} дней (из {max_encoder_length})")
    
#     short_series_count = len(group_stats[group_stats['available_length'] < adaptive_encoder_length + prediction_length])
#     print(f"Групп с короткими рядами (<{adaptive_encoder_length + prediction_length} дней): {short_series_count}")
    
#     # Убеждаемся в правильности типов
#     categorical_cols = ["promoFlag", "is_holiday_day", "weekday", "day", "month", "year",
#                        "is_friday", "is_saturday", "is_sunday", "is_monday",
#                        "is_end_of_month", "is_beginning_of_month", "is_salary_week",
#                        "Магазин", "НоменклатураКод", "GAP"]
#     for col in categorical_cols:
#         if col in df.columns and df[col].dtype.name != 'category':
#             df[col] = df[col].astype(str).astype('category')
    
#     try:
#         # УПРОЩЕННАЯ ВЕРСИЯ: используем только базовые признаки
#         static_categoricals = ["Магазин"]
#         time_varying_known_categoricals = ["promoFlag", "is_holiday_day", "weekday", "day"]
#         time_varying_unknown_reals = ["Количество"]

#         # Оставляем только существующие колонки
#         static_categoricals = [col for col in static_categoricals if col in df.columns]
#         time_varying_known_categoricals = [col for col in time_varying_known_categoricals if col in df.columns]
#         time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]

#         print(f"Используемые признаки:")
#         print(f"  Статические категориальные: {static_categoricals}")
#         print(f"  Известные категориальные: {time_varying_known_categoricals}")
#         print(f"  Неизвестные числовые: {time_varying_unknown_reals}")

#         # Создаем датасет с минимальным набором признаков
#         dataset = TimeSeriesDataSet(
#             df,
#             time_idx="time_idx",
#             target="Количество",
#             group_ids=["Магазин"],
#             min_encoder_length=1,  # УМЕНЬШЕНО для коротких рядов
#             max_encoder_length=adaptive_encoder_length,
#             min_prediction_length=prediction_length,
#             max_prediction_length=prediction_length,
#             static_categoricals=static_categoricals,
#             time_varying_known_categoricals=time_varying_known_categoricals,
#             time_varying_known_reals=["time_idx"],
#             time_varying_unknown_reals=time_varying_unknown_reals,
#             add_relative_time_idx=True,
#             add_target_scales=True,
#             add_encoder_length=True,
#             allow_missing_timesteps=True,
#             # categorical_encoders=categorical_encoders,  # Убираем кастомные энкодеры
#         )
#         return dataset
#     except Exception as e:
#         print(f"Ошибка создания адаптивного датасета: {e}")
        
#         # Резервный вариант: максимально упрощенный датасет
#         print("Пробую максимально упрощенный вариант...")
#         try:
#             # Используем только самые важные признаки
#             simple_df = df[['Магазин', 'time_idx', 'Количество']].copy()
            
#             dataset = TimeSeriesDataSet(
#                 simple_df,
#                 time_idx="time_idx",
#                 target="Количество",
#                 group_ids=["Магазин"],
#                 min_encoder_length=1,
#                 max_encoder_length=adaptive_encoder_length,
#                 min_prediction_length=prediction_length,
#                 max_prediction_length=prediction_length,
#                 static_categoricals=["Магазин"],
#                 time_varying_known_reals=["time_idx"],
#                 time_varying_unknown_reals=["Количество"],
#                 add_relative_time_idx=True,
#                 add_target_scales=True,
#                 add_encoder_length=True,
#                 allow_missing_timesteps=True,
#             )
#             return dataset
#         except Exception as e2:
#             print(f"Ошибка в упрощенной конфигурации: {e2}")
#             raise

# def train_with_all_data(train_ds, val_ds, test_ds, prediction_length, horizon_params):
#     """Обучение с использованием всех данных без фильтрации"""
    
#     try:
#         # Используем меньший батч для обработки неравномерных данных
#         adaptive_batch_size = max(4, horizon_params["batch_size"] // 2)
        
#         train_dl = make_dataloader(train_ds, train=True, batch_size=adaptive_batch_size)
        
#         val_dl = None
#         if val_ds is not None and len(val_ds) > 0:
#             val_dl = make_dataloader(val_ds, train=False, batch_size=adaptive_batch_size * 2)
        
#         test_dl = None  
#         if test_ds is not None and len(test_ds) > 0:
#             test_dl = make_dataloader(test_ds, train=False, batch_size=adaptive_batch_size * 2)

#         # Настройка модели с учетом неравномерности данных - ИСПОЛЬЗУЕМ СТАНДАРТНЫЙ TemporalFusionTransformer
#         model = TemporalFusionTransformer.from_dataset(
#             train_ds,
#             learning_rate=horizon_params["learning_rate"] * 0.5,  # Меньший LR для стабильности
#             hidden_size=horizon_params["hidden_size"],
#             lstm_layers=horizon_params["lstm_layers"],
#             attention_head_size=horizon_params["attention_head_size"],
#             dropout=horizon_params["dropout"] + 0.1,  # Больше регуляризации
#             hidden_continuous_size=horizon_params["hidden_continuous_size"],
#             output_size=1,
#             loss=SMAPE(),
#             reduce_on_plateau_patience=5,  # Больше терпения
#         )
        
#         # Callbacks для работы с неравномерными данными
#         callbacks = []
#         if val_dl is not None:
#             early_stop = EarlyStopping(
#                 monitor="val_loss", 
#                 patience=10,  # Больше терпения
#                 mode="min", 
#                 min_delta=0.001  # Более чувствительный критерий
#             )
#             callbacks.append(early_stop)
        
#         trainer = Trainer(
#             max_epochs=horizon_params["max_epochs"] * 2,  # Больше эпох
#             accelerator="gpu" if USE_GPU else "cpu",
#             devices=1,
#             callbacks=callbacks,
#             precision=precision,
#             accumulate_grad_batches=horizon_params["accumulate_grad_batches"],
#             gradient_clip_val=horizon_params["gradient_clip_val"],
#             check_val_every_n_epoch=2,  # Реже валидация
#             num_sanity_val_steps=0,  # ОТКЛЮЧАЕМ SANITY CHECK
#         )
        
#         print("Обучение на всех данных (включая короткие ряды)...")
        
#         # Обучение с обработкой ошибок
#         try:
#             if val_dl is not None:
#                 trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
#             else:
#                 trainer.fit(model, train_dataloaders=train_dl)
#         except Exception as e:
#             logging.error(f"Ошибка обучения со всеми данными: {e}")
#             return None, None, None
        
#         return model, val_dl, test_dl
        
#     except Exception as e:
#         logging.error(f"Ошибка обучения со всеми данными: {e}")
#         return None, None, None

# #--------------------------------- 
# # ПРОВЕРКА НА КОЛИЧЕСТВО ДАННЫХ:
# # def check_data_sufficiency(df_train, df_val, df_test, max_encoder_length, prediction_length):
# #     min_required_length = max_encoder_length + prediction_length
    
# #     print(f"Минимальная требуемая длина последовательности: {min_required_length}")
    
# #     # Проверяем для каждой группы отдельно
# #     all_groups = set()
# #     if len(df_train) > 0:
# #         all_groups.update(df_train["Магазин"].unique())
# #     if len(df_val) > 0:
# #         all_groups.update(df_val["Магазин"].unique())
# #     if len(df_test) > 0:
# #         all_groups.update(df_test["Магазин"].unique())
    
# #     insufficient_groups = []
    
# #     for group in all_groups:
# #         # Объединяем все данные для группы
# #         group_data = pd.DataFrame()
# #         for df_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
# #             if len(df) > 0 and group in df["Магазин"].values:
# #                 group_data = pd.concat([group_data, df[df["Магазин"] == group]])
        
# #         if len(group_data) == 0:
# #             continue
            
# #         min_time = group_data["time_idx"].min()
# #         max_time = group_data["time_idx"].max()
# #         available_length = max_time - min_time + 1
# #         data_points = len(group_data)
        
# #         print(f"Группа {group}: временной диапазон {min_time}-{max_time}, "
# #               f"доступная длина {available_length}, точек данных {data_points}")
        
# #         if available_length < min_required_length:
# #             insufficient_groups.append({
# #                 'group': group,
# #                 'available': available_length,
# #                 'required': min_required_length,
# #                 'data_points': data_points
# #             })
    
# #     if insufficient_groups:
# #         print("Группы с недостаточными данными:")
# #         for group_info in insufficient_groups:
# #             print(f"  {group_info['group']}: доступно {group_info['available']}, "
# #                   f"требуется {group_info['required']}, точек данных {group_info['data_points']}")
    
# #     return insufficient_groups


# def check_indices(df):
#     """Проверка целостности индексов"""
#     issues = []
#     for group_name, group_data in df.groupby("Магазин"):
#         # Сортируем по time_idx
#         group_data = group_data.sort_values("time_idx")
        
#         # Проверка непрерывности time_idx
#         time_diff = np.diff(sorted(group_data["time_idx"].unique()))
#         if len(time_diff) > 0 and np.any(time_diff > 1):
#             gaps = time_diff[time_diff > 1]
#             issues.append(f"Пропуски в time_idx у {group_name}: {len(gaps)} пропусков")
        
#         # Проверка уникальности time_idx внутри групп
#         duplicate_mask = group_data.duplicated(subset=["НоменклатураКод", "time_idx"], keep=False)
#         if duplicate_mask.any():
#             duplicate_count = duplicate_mask.sum()
#             issues.append(f"Дубликаты time_idx у {group_name}: {duplicate_count} дубликатов")
            
#             # ИСПРАВЛЕНИЕ: удаляем дубликаты, оставляя первую запись
#             print(f"Удаляем {duplicate_count} дубликатов в группе {group_name}")
#             df = df.drop_duplicates(subset=["Магазин", "НоменклатураКод", "time_idx"], keep='first')
    
#     return issues, df

# # -----------------------------------------------------------------------------
# # ОБУЧЕНИЕ С НАКОПЛЕНИЕМ ГРАДИЕНТОВ И ЧЕКПОИНТАМИ
# # -----------------------------------------------------------------------------
# # def check_dataloader(dataloader, name="train"):
# #     """Проверка DataLoader перед обучением"""
# #     try:
# #         print(f"Проверка {name} DataLoader...")
# #         batch = next(iter(dataloader))
# #         print(f"  Тип батча: {type(batch)}")
# #         if isinstance(batch, (list, tuple)):
# #             print(f"  Длина батча: {len(batch)}")
# #             for i, item in enumerate(batch):
# #                 print(f"    Элемент {i}: тип={type(item)}")
# #         print(f"  {name} DataLoader готов")
# #         return True
# #     except Exception as e:
# #         print(f"  Ошибка в {name} DataLoader: {e}")
# #         return False
    

# # def train_one_horizon(train_ds, val_ds, test_ds, prediction_length: int, horizon_params: dict):
# #     """Обучение модели с мониторингом и защитой"""
# #     try:
# #         if train_ds is None or len(train_ds) == 0:
# #             logging.error("Пустой тренировочный датасет")
# #             return None, None, None
        
# #         # try:
# #         #     sample = next(iter(train_ds))
# #         #     print(f"Пример данных из датасета: {type(sample)}")
# #         # except Exception as e:
# #         #     print(f"Ошибка при проверке датасета: {e}")
# #         #     return None, None, None
        
# #         train_dl = make_dataloader(train_ds, train=True, batch_size=horizon_params["batch_size"])
# #         # Проверяем, что DataLoader не пустой
# #         # try:
# #         #     test_batch = next(iter(train_dl))
# #         #     logging.info(f"Размер батча: {test_batch[0].shape if isinstance(test_batch, (list, tuple)) else 'unknown'}")
# #         # except StopIteration:
# #         #     logging.error("Train DataLoader пустой")
# #         #     return None, None, None
        
# #         val_dl = None
# #         if val_ds is not None and len(val_ds) > 0:
# #             try:
# #                 val_dl = make_dataloader(val_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
# #             except Exception as e:
# #                 print(f"Ошибка создания val_dataloader: {e}")
# #                 val_dl = None
        
# #         test_dl = None
# #         if test_ds is not None and len(test_ds) > 0:
# #             try:
# #                 test_dl = make_dataloader(test_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
# #             except Exception as e:
# #                 print(f"Ошибка создания test_dataloader: {e}")
# #                 test_dl = None

# #         # val_dl = make_dataloader(val_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
# #         # test_dl = make_dataloader(test_ds, train=False, batch_size=horizon_params["batch_size"] * 2)

# #         print(f"--- Параметры для горизонта {prediction_length} дней:")
# #         print(f"   Batch size: {horizon_params['batch_size']}")
# #         print(f"   Train batches: {len(train_dl) if train_dl else 0}")
# #         print(f"   Val batches: {len(val_dl) if val_dl else 0}")

# #         model = FixedTFT.from_dataset(
# #             train_ds,
# #             learning_rate=horizon_params["learning_rate"],
# #             hidden_size=horizon_params["hidden_size"],
# #             lstm_layers=horizon_params["lstm_layers"],
# #             attention_head_size=horizon_params["attention_head_size"],
# #             dropout=horizon_params["dropout"],
# #             hidden_continuous_size=horizon_params["hidden_continuous_size"],
# #             output_size=1,
# #             loss=SMAPE(),
# #             # log_interval=100,
# #             # log_val_interval=100,
# #             reduce_on_plateau_patience=3,
# #             # optimizer="adamw",
# #             # weight_decay=1e-5,
# #         )

# #         # Ускорение на torch>=2.0
# #         # try:
# #         #     model = torch.compile(model)
# #         # except Exception:
# #         #     pass

# #         # УЛУЧШЕННЫЕ ЧЕКПОИНТЫ
# #         # checkpoint_dir = os.path.join(MODEL_DIR, f"horizon_{prediction_length}")
# #         # os.makedirs(checkpoint_dir, exist_ok=True)
        
# #         early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min") if val_dl else None  #,stopping_threshold=0.3)
    
# #         # ckpt = ModelCheckpoint(
# #         #     dirpath=checkpoint_dir,
# #         #     filename=f"tft_{prediction_length}d-{{epoch}}-{{val_loss:.3f}}",
# #         #     monitor="val_loss",
# #         #     save_top_k=1,  # Сохраняем 2 лучшие модели
# #         #     mode="min",
# #         #     # save_last=False,  # Отключаем сохранение последней
# #         #     # every_n_epochs=1,  # Сохраняем каждую эпоху
# #         # )
    
# #         lr_logger = LearningRateMonitor(logging_interval="step")

# #         callbacks = [lr_logger]
# #         if early_stop:
# #             callbacks.append(early_stop)

# #         # ТРЕНЕР С ЗАЩИТОЙ
# #         trainer = Trainer(
# #             max_epochs=1,  #horizon_params["max_epochs"],
# #             limit_train_batches=horizon_params["limit_train_batches"],
# #             limit_val_batches=horizon_params["limit_val_batches"] if val_dl is not None else 0,
# #             accelerator="gpu" if USE_GPU else "cpu",
# #             devices=1,  # Явно указываем 1 устройство
# #             callbacks=callbacks,  #early_stop, lr_logger, ckpt],
# #             precision=precision,  # Автоматический выбор
# #             accumulate_grad_batches=1,  #horizon_params["accumulate_grad_batches"],
# #             gradient_clip_val=horizon_params["gradient_clip_val"], 
# #             # gradient_clip_algorithm="norm",
# #             log_every_n_steps=10,  #50,
# #             check_val_every_n_epoch=1,  #1,
# #             enable_progress_bar=True,
# #             num_sanity_val_steps=2 if val_dl else 0,  # ОТКЛЮЧАЕМ SANITY CHECK
# #             # deterministic=True,  # Для воспроизводимости
# #         )

# #         print("=== Начинаю обучение...")
# #         print_memory_usage()
        
# #         t0 = time.time()
# #         # Обучение с обработкой ошибок
# #         try:
# #             if val_dl is not None:
# #                 trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
# #             else:
# #                 trainer.fit(model, train_dataloaders=train_dl)
# #         except Exception as e:
# #             logging.error(f"Ошибка обучения: {e}")
# #             return None, None, None
        
# #         training_time = (time.time() - t0) / 60
# #         logging.info(f"Время обучения (h{prediction_length}): {training_time:.1f} минут")
# #         print(f"Время обучения (h{prediction_length}): {training_time:.1f} минут")

# #         # # Загружаем лучшую модель из чекпоинтов
# #         # if ckpt.best_model_path and os.path.exists(ckpt.best_model_path):
# #         #     logging.info(f"Загружаю лучшую модель: {ckpt.best_model_path}")
# #         #     print(f"Загружаю лучшую модель: {ckpt.best_model_path}")
# #         #     try:
# #         #         model = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
# #         #     except Exception as e:
# #         #         logging.error(f"Ошибка загрузки модели: {e}")
# #         # else:
# #         #     logging.warning("Лучший чекпоинт не найден, использую последнюю модель")

# #         return model, val_dl, test_dl
# #     except Exception as e:
# #         error_logger.error(f"Ошибка в train_one_horizon: {e}")
# #         return None, None, None

# def train_one_horizon(train_ds, val_ds, test_ds, prediction_length: int, horizon_params: dict):
#     """Обучение модели с мониторингом и защитой"""
#     try:
#         if train_ds is None or len(train_ds) == 0:
#             logging.error("Пустой тренировочный датасет")
#             return None, None, None
        
#         train_dl = make_dataloader(train_ds, train=True, batch_size=horizon_params["batch_size"])
        
#         val_dl = None
#         if val_ds is not None and len(val_ds) > 0:
#             try:
#                 val_dl = make_dataloader(val_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
#             except Exception as e:
#                 print(f"Ошибка создания val_dataloader: {e}")
#                 val_dl = None
        
#         test_dl = None
#         if test_ds is not None and len(test_ds) > 0:
#             try:
#                 test_dl = make_dataloader(test_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
#             except Exception as e:
#                 print(f"Ошибка создания test_dataloader: {e}")
#                 test_dl = None

#         print(f"--- Параметры для горизонта {prediction_length} дней:")
#         print(f"   Batch size: {horizon_params['batch_size']}")
#         print(f"   Train batches: {len(train_dl) if train_dl else 0}")
#         print(f"   Val batches: {len(val_dl) if val_dl else 0}")

#         # ИСПОЛЬЗУЕМ СТАНДАРТНЫЙ TemporalFusionTransformer БЕЗ КАСТОМНОГО КЛАССА
#         model = TemporalFusionTransformer.from_dataset(
#             train_ds,
#             learning_rate=horizon_params["learning_rate"],
#             hidden_size=horizon_params["hidden_size"],
#             lstm_layers=horizon_params["lstm_layers"],
#             attention_head_size=horizon_params["attention_head_size"],
#             dropout=horizon_params["dropout"],
#             hidden_continuous_size=horizon_params["hidden_continuous_size"],
#             output_size=1,
#             loss=SMAPE(),
#             reduce_on_plateau_patience=3,
#         )

#         # ИСПРАВЛЕНИЕ: early stopping только если есть валидационные данные
#         callbacks = []
#         if val_dl is not None:
#             early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
#             callbacks.append(early_stop)
        
#         lr_logger = LearningRateMonitor(logging_interval="step")
#         callbacks.append(lr_logger)

#         # УПРОЩЕННЫЙ ТРЕНЕР
#         trainer = Trainer(
#             max_epochs=1,
#             limit_train_batches=0.01,  # ЕЩЕ МЕНЬШЕ ДЛЯ ТЕСТА
#             limit_val_batches=1.0 if val_dl else 0,
#             accelerator="gpu" if USE_GPU else "cpu",
#             devices=1,
#             callbacks=callbacks,
#             precision=precision,
#             accumulate_grad_batches=1,
#             gradient_clip_val=horizon_params["gradient_clip_val"], 
#             log_every_n_steps=10,
#             check_val_every_n_epoch=1,
#             enable_progress_bar=True,
#             num_sanity_val_steps=0,  # ОТКЛЮЧАЕМ SANITY CHECK
#         )

#         print("=== Начинаю обучение...")
#         print_memory_usage()
        
#         t0 = time.time()
#         # Обучение с обработкой ошибок
#         try:
#             if val_dl is not None:
#                 trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
#             else:
#                 trainer.fit(model, train_dataloaders=train_dl)
#         except Exception as e:
#             logging.error(f"Ошибка обучения: {e}")
#             print(f"Детали ошибки: {type(e).__name__}: {e}")
#             return None, None, None
        
#         training_time = (time.time() - t0) / 60
#         logging.info(f"Время обучения (h{prediction_length}): {training_time:.1f} минут")
#         print(f"Время обучения (h{prediction_length}): {training_time:.1f} минут")

#         return model, val_dl, test_dl
#     except Exception as e:
#         error_logger.error(f"Ошибка в train_one_horizon: {e}")
#         print(f"Критическая ошибка в train_one_horizon: {type(e).__name__}: {e}")
#         return None, None, None
    
# # -----------------------------------------------------------------------------
# # Оценка 
# # -----------------------------------------------------------------------------
# def evaluate_metrics(model, dataloader):
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     try:
#         all_preds = []
#         all_targets = []
        
#         model.eval()
#         with torch.no_grad():
#             for batch in dataloader:
#                 # ИСПРАВЛЕНИЕ: правильная обработка формата батча
#                 if isinstance(batch, (list, tuple)) and len(batch) == 2:
#                     # Батч в формате (x, y) 
#                     x, y = batch
#                     output = model(x)  # Передаем только x
#                     pred = output.prediction
                    
#                     # Получаем целевые значения из y
#                     # y может быть списком [decoder_target, target]
#                     if isinstance(y, list) and len(y) >= 1:
#                         targets = y[0]  # Берем decoder_target
#                     else:
#                         targets = y
#                 else:
#                     # Батч в формате словаря
#                     output = model(batch)
#                     pred = output.prediction
#                     if isinstance(batch, dict):
#                         targets = batch.get("decoder_target", batch.get("target", None))
#                     else:
#                         targets = batch[1] if len(batch) > 1 else None
                
#                 # Проверяем, что pred и targets не None
#                 if pred is None or targets is None:
#                     print("Предупреждение: pred или targets None")
#                     continue
                
#                 # Перемещаем на CPU и преобразуем в numpy
#                 pred_np = pred.cpu().numpy()
#                 targets_np = targets.cpu().numpy()
                
#                 # Изменяем форму если нужно
#                 if pred_np.ndim > 2:
#                     pred_np = pred_np.reshape(pred_np.shape[0], -1)
#                 if targets_np.ndim > 2:
#                     targets_np = targets_np.reshape(targets_np.shape[0], -1)
                
#                 all_preds.append(pred_np)
#                 all_targets.append(targets_np)
        
#         if not all_preds or not all_targets:
#             print("Нет данных для оценки")
#             return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        
#         # Объединяем результаты
#         preds = np.concatenate(all_preds, axis=0)
#         y_true = np.concatenate(all_targets, axis=0)
        
#         # Flatten arrays для метрик
#         preds_flat = preds.flatten()
#         y_true_flat = y_true.flatten()
        
#         # Удаляем NaN и Inf значения
#         mask = np.isfinite(preds_flat) & np.isfinite(y_true_flat)
#         preds_flat = preds_flat[mask]
#         y_true_flat = y_true_flat[mask]
        
#         if len(preds_flat) == 0:
#             print("Нет валидных данных для расчета метрик")
#             return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        
#         # Расчет метрик
#         mae = mean_absolute_error(y_true_flat, preds_flat)
#         rmse = mean_squared_error(y_true_flat, preds_flat, squared=False)
#         r2 = r2_score(y_true_flat, preds_flat)
        
#         # SMAPE и MAPE только для ненулевых значений
#         non_zero_mask = y_true_flat != 0
#         if np.any(non_zero_mask):
#             y_true_nonzero = y_true_flat[non_zero_mask]
#             y_pred_nonzero = preds_flat[non_zero_mask]
#             smape_val = 100 * np.mean(2 * np.abs(y_pred_nonzero - y_true_nonzero) / (np.abs(y_true_nonzero) + np.abs(y_pred_nonzero)))
#             mape = 100 * np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero))
#         else:
#             smape_val = float('nan')
#             mape = float('nan')
        
#         return mae, rmse, r2, smape_val, mape
        
#     except Exception as e:
#         print(f"Ошибка при оценке модели: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
# # def check_data_quality(df):
# #     """Проверка качества данных перед обработкой"""
# #     print("=== ПРОВЕРКА КАЧЕСТВА ДАННЫХ ===")
    
# #     # Проверяем основные колонки
# #     critical_columns = ['Дата', 'Магазин', 'НоменклатураКод', 'Количество']
# #     for col in critical_columns:
# #         if col in df.columns:
# #             nan_count = df[col].isnull().sum()
# #             unique_count = df[col].nunique()
# #             print(f"{col}: {nan_count} NaN, {unique_count} уникальных значений")
    
# #     # Проверяем даты
# #     if 'Дата' in df.columns:
# #         date_stats = df['Дата'].describe()
# #         print(f"Даты: от {date_stats['first']} до {date_stats['last']}")
        
# #         # Проверяем некорректные даты
# #         invalid_dates = df[df['Дата'].isnull()]
# #         if len(invalid_dates) > 0:
# #             print(f"Найдено {len(invalid_dates)} некорректных дат")
    
# #     # Проверяем числовые колонки
# #     numeric_cols = ['Количество', 'Сумма', 'Себестоимость', 'marginCost']
# #     for col in numeric_cols:
# #         if col in df.columns:
# #             stats = df[col].describe()
# #             print(f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")


# # Добавьте эту временную функцию для самого простого теста
# def quick_test():
#     """Быстрый тест на минимальных данных"""
#     print("=== БЫСТРЫЙ ТЕСТ ===")
    
#     # Сначала получаем доступные магазины и категории
#     available_shops, available_gaps = get_available_shops_and_gaps(DB_PATH)
    
#     # Берем только первый магазин и первую категорию
#     test_shops = available_shops[:1]
#     test_gaps = available_gaps[:1]
    
#     print(f"Тестовые данные: {test_shops[0]}, {test_gaps[0]}")
    
#     # Загружаем минимальные данные
#     df_test = prepare_data_from_db(DB_PATH, test_shops, test_gaps)
#     df_test = aggregate_daily_data(df_test)
    
#     # Простое разделение
#     df_train, df_val, df_test = split_by_time(df_test)
    
#     print(f"Тест готов: Train={len(df_train)}, Val={len(df_val)}")
    
#     # Пробуем создать датасет и обучить
#     horizon = 30
#     horizon_params = HORIZON_PARAMS[horizon]
    
#     train_ds = create_adaptive_dataset(df_train, prediction_length=horizon)
#     val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True) if len(df_val) > 0 else None
    
#     model, val_dl, _ = train_one_horizon(train_ds, val_ds, None, horizon, horizon_params)
    
#     if model is not None:
#         print("Тест пройден! Обучение работает.")
#     else:
#         print("Тест не пройден.")
    
#     return model

# def analyze_missing_data(df):
#     """Анализ пропущенных данных"""
#     print("\n=== АНАЛИЗ ПРОПУЩЕННЫХ ДАННЫХ ===")
    
#     # Пропуски по колонкам
#     missing_data = df.isnull().sum()
#     missing_percent = (missing_data / len(df)) * 100
    
#     missing_info = pd.DataFrame({
#         'Пропусков': missing_data,
#         'Процент': missing_percent
#     })
    
#     # Только колонки с пропусками
#     missing_info = missing_info[missing_info['Пропусков'] > 0]
    
#     if len(missing_info) > 0:
#         print("Колонки с пропусками:")
#         print(missing_info.sort_values('Пропусков', ascending=False))
#     else:
#         print("Пропусков нет!")
    
#     # Анализ нулевых продаж
#     zero_sales = (df['Количество'] == 0).sum()
#     print(f"\nДней с нулевыми продажами: {zero_sales} ({zero_sales/len(df)*100:.1f}%)")
    
#     return missing_info


# # -----------------------------------------------------------------------------
# # MAIN С ОЧИСТКОЙ ПАМЯТИ
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     optimize_memory()

#     # Настройка перехвата ошибок
#     sys.excepthook = lambda exctype, value, traceback: error_logger.error(
#         f"Необработанное исключение: {exctype.__name__}: {value}", exc_info=True
#     )
    
#     logging.info("Загрузка данных...")
    
#     # Сначала получаем доступные магазины и категории
#     available_shops, available_gaps = get_available_shops_and_gaps(DB_PATH)

#     # Теперь запускаем быстрый тест
#     # quick_test()

#     # ОПТИМИЗАЦИЯ: ограничиваем данные для тестирования
#     TEST_MODE = True

#     # Настройка перехвата ошибок
#     sys.excepthook = lambda exctype, value, traceback: error_logger.error(
#         f"Необработанное исключение: {exctype.__name__}: {value}", exc_info=True
#     )
    
#     logging.info("Загрузка данных...")
    
#     available_shops, available_gaps = get_available_shops_and_gaps(DB_PATH)

#     # Выбираем магазины и категории для обучения (можно настроить фильтрацию)
#     if TEST_MODE:
#         # Берем только 2 магазина и 3 категории для теста
#         selected_shops = available_shops[:2] if len(available_shops) > 2 else available_shops
#         selected_gaps = available_gaps[:3] if len(available_gaps) > 3 else available_gaps
#         print(f"ТЕСТОВЫЙ РЕЖИМ: {len(selected_shops)} магазинов, {len(selected_gaps)} категорий")
#     else:
#         selected_shops = available_shops
#         selected_gaps = available_gaps

#     df = prepare_data_from_db(DB_PATH, selected_shops, selected_gaps)  # load_raw()
#     # В main после prepare_data_from_db добавьте:
#     print("=== ФИНАЛЬНАЯ ПРОВЕРКА ТИПОВ ДАННЫХ ===")
#     categorical_check = ['weekday', 'day', 'promoFlag', 'is_holiday_day', 'Месяц', 'ГОД']
#     for col in categorical_check:
#         if col in df.columns:
#             dtype = df[col].dtype
#             sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
#             print(f"{col}: тип={dtype}, пример={sample}, тип_значения={type(sample)}")
            
#             # Принудительное исправление если нужно
#             if dtype.name != 'category':
#                 print(f"Исправляем: {col} -> category")
#                 df[col] = df[col].astype('category')
#             elif not isinstance(sample, str) and sample is not None:
#                 print(f"Исправляем значения: {col} -> str -> category")
#                 df[col] = df[col].astype(str).astype('category')
#     shop_stats = check_shop_data_distribution(df)
#     print(shop_stats)
#     # check_data_quality(df)
#     # Агрегируем данные по дням для каждого товара в каждом магазине
#     df = aggregate_daily_data(df)

#     print("=== ДИАГНОСТИКА ПЕРЕД СОЗДАНИЕМ DATASET ===")
#     print("Типы данных перед созданием dataset:")
#     for col in ['weekday', 'day', 'promoFlag', 'is_holiday_day']:
#         if col in df.columns:
#             dtype = df[col].dtype
#             sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
#             print(f"  {col}: {dtype}, пример: {sample}, тип значения: {type(sample)}")

#     # Принудительное исправление если нужно
#     for col in ['weekday', 'day', 'promoFlag', 'is_holiday_day']:
#         if col in df.columns:
#             if df[col].dtype.name != 'category':
#                 print(f"Принудительно преобразуем {col} в категорию")
#                 df[col] = df[col].astype('category')

#     print("=== ПРОВЕРКА СТРУКТУРЫ ДАННЫХ ===")
#     print(f"Столбцы после агрегации: {df.columns.tolist()}")
#     print(f"Размер данных: {len(df)}")
#     required_columns = ['Магазин', 'НоменклатураКод', 'time_idx', 'Дата', 'Количество']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         print(f"ОШИБКА: Отсутствуют обязательные столбцы: {missing_columns}")
#         sys.exit(1)
#     # time_analysis = analyze_time_points(df)

#     df_train, df_val, df_test = split_by_time(df)

#     print("=== ПРОВЕРКА РАЗДЕЛЕНИЯ ДАННЫХ ===")
#     print(f"Train размер: {len(df_train)}")
#     print(f"Val размер: {len(df_val)}") 
#     print(f"Test размер: {len(df_test)}")

#     if len(df_train) == 0:
#         raise ValueError("Нет данных для обучения!")

#     print("Группы в train:", df_train["Магазин"].unique())
#     print("Группы в val:", df_val["Магазин"].unique() if len(df_val) > 0 else "Нет данных")
#     print("Группы в test:", df_test["Магазин"].unique() if len(df_test) > 0 else "Нет данных")

#     # Проверяем, что все группы из val и test есть в train
#     train_groups = set(df_train["Магазин"].unique())
#     if len(df_val) > 0:
#         val_groups = set(df_val["Магазин"].unique())
#         missing_in_train = val_groups - train_groups
#         if missing_in_train:
#             print(f"⚠️  Предупреждение: в val есть группы, отсутствующие в train: {missing_in_train}")
#             print("Удаляем эти группы из val...")
#             df_val = df_val[df_val["Магазин"].isin(train_groups)]

#     if len(df_test) > 0:
#         test_groups = set(df_test["Магазин"].unique())
#         missing_in_train = test_groups - train_groups
#         if missing_in_train:
#             print(f"⚠️  Предупреждение: в test есть группы, отсутствующие в train: {missing_in_train}")
#             print("Удаляем эти группы из test...")
#             df_test = df_test[df_test["Магазин"].isin(train_groups)]

#     print("После фильтрации:")
#     print(f"Train: {len(df_train)} записей, группы: {df_train['Магазин'].unique()}")
#     print(f"Val: {len(df_val)} записей, группы: {df_val['Магазин'].unique() if len(df_val) > 0 else 'Нет'}")
#     print(f"Test: {len(df_test)} записей, группы: {df_test['Магазин'].unique() if len(df_test) > 0 else 'Нет'}")


#     print("=== ПРОВЕРКА ТИПОВ ДАННЫХ ===")
#     print("Типы данных в train:")
#     print(df_train.dtypes)

#     # Проверяем конкретно проблемные колонки
#     problem_columns = ['promoFlag', 'is_holiday_day', 'Месяц', 'ГОД', 'weekday', 'day']
#     for col in problem_columns:
#         if col in df_train.columns:
#             print(f"{col}: тип={df_train[col].dtype}, примеры={df_train[col].unique()[:5]}")

#     df_train, df_val, df_test, scaler = scale_numeric(df_train, df_val, df_test)

#     print("Проверка целостности данных...")
#     print(f"Всего записей: {len(df)}")
#     print(f"Уникальных магазинов: {df['Магазин'].nunique()}")
#     print(f"Диапазон дат: {df['Дата'].min()} - {df['Дата'].max()}")
#     print(f"Диапазон time_idx: {df['time_idx'].min()} - {df['time_idx'].max()}")

#     # Проверка на пропущенные значения
#     for col in df.columns:
#         if df[col].isnull().sum() > 0:
#             print(f"Пропуски в {col}: {df[col].isnull().sum()}")

#     # Проверка индексов
#     index_issues, df_train = check_indices(df_train)
#     if index_issues:
#         print("Проблемы с индексами:")
#         for issue in index_issues:
#             print(f"  - {issue}")

#     results = []

#     for horizon in HORIZONS:
#         print(f"\n{'='*60}")
#         print(f"=======ОБУЧЕНИЕ ДЛЯ ГОРИЗОНТА {horizon} ДНЕЙ")
#         print(f"{'='*60}")
        
#         # Очистка памяти перед каждым обучением
#         clear_memory()
        
#         # Получаем параметры для горизонта
#         horizon_params = HORIZON_PARAMS[horizon]
#         logging.info(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
#         print(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        
#         # АНАЛИЗ ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ
#         print("=== АНАЛИЗ ДОСТУПНЫХ ДАННЫХ ===")
#         total_groups = df['Магазин'].nunique()
#         total_items = df['НоменклатураКод'].nunique()
        
#         group_stats = df.groupby('Магазин')['time_idx'].agg(['min', 'max', 'count']).reset_index()
#         group_stats['length'] = group_stats['max'] - group_stats['min'] + 1
        
#         print(f"Всего групп: {total_groups}")
#         print(f"Всего товаров: {total_items}")
#         print(f"Средняя длина ряда: {group_stats['length'].mean():.1f} дней")
#         print(f"Минимальная длина: {group_stats['length'].min()} дней")
#         print(f"Максимальная длина: {group_stats['length'].max()} дней")

#         # # Детальный анализ временных рядов
#         # ts_analysis = analyze_group_timeseries(df)
        
#         # Разделяем данные для этого горизонта (БЕЗ ФИЛЬТРАЦИИ)
#         print("=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
#         df_train, df_val, df_test = split_by_time(df)
        
#         logging.info(f"Данные для горизонта {horizon}: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
#         print(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

#         # # проверка на количество данных
#         # min_required = horizon_params["max_encoder_length"] + horizon
#         # logging.info(f"Требуется минимум {min_required} временных точек")
#         # print(f"Требуется минимум {min_required} временных точек")
#         # check_data_sufficiency(df_train, df_val, df_test, horizon_params["max_encoder_length"], horizon)

#         # Масштабирование
#         df_train, df_val, df_test, scaler = scale_numeric(df_train, df_val, df_test)

#         # Создаем datasets
#         try:
#             print("=== СОЗДАНИЕ АДАПТИВНОГО ДАТАСЕТА ===")
#             train_ds = create_adaptive_dataset(df_train, prediction_length=horizon, 
#                                             max_encoder_length=horizon_params["max_encoder_length"])
            
#             val_ds = None
#             if len(df_val) > 0:
#                 val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True)
            
#             test_ds = None
#             if len(df_test) > 0:
#                 test_ds = TimeSeriesDataSet.from_dataset(train_ds, df_test, predict=True)
                
#         except Exception as e:
#             print(f"Ошибка создания адаптивного датасета: {e}")
#             # Пробуем стандартный подход как запасной вариант
#             try:
#                 print("Пробуем стандартное создание датасета...")
#                 train_ds = make_dataset(df_train, prediction_length=horizon)
#                 val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True) if len(df_val) > 0 else None
#                 test_ds = TimeSeriesDataSet.from_dataset(train_ds, df_test, predict=True) if len(df_test) > 0 else None
#             except Exception as e2:
#                 print(f"Ошибка в стандартном подходе: {e2}")
#                 continue

#         # Обучение
#         model, val_dl, test_dl = train_one_horizon(train_ds, val_ds, test_ds, prediction_length=horizon, horizon_params=horizon_params)
        
#         if model is None:
#             print("Стандартное обучение не удалось, пробуем адаптивный подход...")
#             model, val_dl, test_dl = train_with_all_data(train_ds, val_ds, test_ds, horizon, horizon_params)
        
#         if model is None:
#             logging.error(f"Обучение для горизонта {horizon} не удалось"