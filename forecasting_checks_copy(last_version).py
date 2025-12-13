import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
import torch
import gc
import logging
import signal
import sys
from datetime import datetime
import sqlite3
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, SMAPE

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import vizualization_forecasting as vz

def optimize_memory():
    """Принудительная очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# -----------------------------------------------------------------------------
# НАСТРОЙКА ЛОГГИРОВАНИЯ И ОБРАБОТКИ ОШИБОК
# -----------------------------------------------------------------------------
def setup_logging():
    """Настройка комплексного логирования - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Основной логгер
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Только консоль, без файла
            ]
        )
        
        # Логгер для ошибок (только консоль)
        error_logger = logging.getLogger('error_logger')
        error_logger.setLevel(logging.ERROR)
        
        print(f"Логирование настроено. Время: {timestamp}")
        
        return error_logger
        
    except Exception as e:
        print(f"Ошибка настройки логирования: {e}")
        return logging.getLogger()

def signal_handler(sig, frame):
    """Обработчик аварийного завершения"""
    error_logger.error("ПРИНУДИТЕЛЬНОЕ ЗАВЕРШЕНИЕ ПОЛЬЗОВАТЕЛЕМ!")
    torch.cuda.empty_cache()
    sys.exit(1)

# -----------------------------------------------------------------------------
# Быстрые настройки PyTorch
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

# Ограничение памяти GPU
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.90)

# USE_GPU = torch.cuda.is_available()
# SUPPORTS_BF16 = USE_GPU and torch.cuda.is_bf16_supported()

# # Автоматическое определение precision. В конечном итоге обучала на 32-true
# if USE_GPU:
#     device_name = torch.cuda.get_device_name(0)
#     if "3090" in device_name:
#         precision = "bf16"
#     else:
#         precision = "32-true"
# else:
precision = "32-true"

PIN_MEMORY = False
NUM_WORKERS = 0

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'sales_data.db')
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Инициализация логирования
error_logger = setup_logging()

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# logging.info(f"GPU available: {USE_GPU}")
# print(f"GPU available: {USE_GPU}")
# if USE_GPU:
#     logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
#     logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# -----------------------------------------------------------------------------
# ГИПЕРПАРАМЕТРЫ
# -----------------------------------------------------------------------------

def get_adaptive_hyperparameters(df_train, prediction_length):
    """Адаптивные гиперпараметры"""
    
    n_samples = len(df_train)
    n_items = df_train['НоменклатураКод'].nunique()
    target_stats = df_train['Количество'].describe()
    
    print(f"Анализ данных: {n_samples} samples, {n_items} items")
    print(f"Статистика целевой переменной: min={target_stats['min']:.1f}, max={target_stats['max']:.1f}, mean={target_stats['mean']:.1f}")
    
    base_config = {
        30: {
            "learning_rate": 0.001,
            "hidden_size": max(16, min(32, n_items // 10)),
            "lstm_layers": 2,
            "attention_head_size": 1,
            "dropout": 0.2,
            "hidden_continuous_size": 8,
            "batch_size": 32,
            "max_encoder_length": 60,
            "max_epochs": 50,
            "patience": 10,
            "gradient_clip_val": 0.1,
        },
        60: {
            "learning_rate": 0.0001,
            "hidden_size": max(16, min(24, n_items // 15)),
            "lstm_layers": 2,
            "attention_head_size": 1,
            "dropout": 0.3,  # Больше регуляризации
            "hidden_continuous_size": 4,
            "batch_size": 8,  # Меньше batch size для стабильности
            "max_encoder_length": 30,  # Согласовано с датасетом
            "max_epochs": 40,
            "patience": 8,
            "gradient_clip_val": 0.5,
        }
    }
    
    config = base_config[prediction_length].copy()
    
    print(f"Адаптивные параметры для {prediction_length}d: LR={config['learning_rate']}, hidden={config['hidden_size']}")
    return config

class DiscreteSalesLoss(nn.Module):
    """Кастомная функция потерь для дискретных продаж"""
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        
    def forward(self, y_pred, y_actual):
        mae_loss = self.mae(y_pred, y_actual)
        
        lower_penalty = F.relu(1.0 - y_pred).mean()
        upper_penalty = F.relu(y_pred - 4.0).mean()
        bound_penalty = lower_penalty + upper_penalty
        
        rounded_pred = torch.round(y_pred)
        integer_penalty = F.mse_loss(y_pred, rounded_pred)
        
        total_loss = mae_loss + self.alpha * (bound_penalty + integer_penalty)
        return total_loss

def create_adaptive_model(train_ds, horizon_params):
    """Создание модели с автоматической настройкой"""
    try:
        # Автоматический подбор LR если не задан
        if "learning_rate" not in horizon_params or horizon_params["learning_rate"] is None:
            base_lr = 0.001
            if horizon_params.get("max_encoder_length", 30) > 50:
                base_lr = 0.0005
            horizon_params["learning_rate"] = base_lr
        
        from pytorch_forecasting.metrics import MAE
        
        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=horizon_params["learning_rate"],
            hidden_size=horizon_params["hidden_size"],
            lstm_layers=horizon_params["lstm_layers"],
            attention_head_size=horizon_params["attention_head_size"],
            dropout=horizon_params["dropout"],
            hidden_continuous_size=horizon_params["hidden_continuous_size"],
            output_size=1,
            loss=MAE(),  # ПРОСТАЯ MAE
            reduce_on_plateau_patience=horizon_params.get("patience", 5),
        )
        return model
    except Exception as e:
        print(f"Ошибка создания модели: {e}")
        return None

HORIZONS = [30, 60] 

# -----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------------------
def clear_memory():
    """Очистка памяти GPU и CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Память очищена")

def print_memory_usage():
    """Вывод информации об использовании памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU память: {allocated:.1f}GB / {reserved:.1f}GB")
    
    import psutil
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")

def aggressive_memory_cleanup():
    """Агрессивная очистка памяти"""
    import gc
    # Многократная очистка
    for i in range(5):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Принудительный вызов сборщика
    if hasattr(gc, 'get_referrers'):
        gc.collect()

# -----------------------------------------------------------------------------
# ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ
# -----------------------------------------------------------------------------
def load_data_from_db(db_path, shops=None, gaps=None, min_date=None, max_date=None):
    """Загрузка данных из SQLite базы данных"""
    print("Загрузка данных из базы данных...")
    
    conn = sqlite3.connect(db_path)
    
    # Сначала получаем количество записей
    count_query = "SELECT COUNT(*) FROM sales_data WHERE 1=1"
    params = []
    
    if shops:
        placeholders = ','.join(['?' for _ in shops])
        count_query += f" AND Магазин IN ({placeholders})"
        params.extend(shops)
    if gaps:
        placeholders = ','.join(['?' for _ in gaps])
        count_query += f" AND GAP IN ({placeholders})"
        params.extend(gaps)
    
    total_rows = pd.read_sql_query(count_query, conn, params=params).iloc[0,0]
    print(f"Всего записей для загрузки: {total_rows}")
    
    query = """
    SELECT Дата, Магазин, НоменклатураКод, GAP, Количество, 
           promoFlag, Месяц, ГОД, is_holiday_day
    FROM sales_data WHERE 1=1
    """
    
    if shops:
        placeholders = ','.join(['?' for _ in shops])
        query += f" AND Магазин IN ({placeholders})"
    if gaps:
        placeholders = ','.join(['?' for _ in gaps])
        query += f" AND GAP IN ({placeholders})"
    
    query += " ORDER BY Магазин, НоменклатураКод, Дата"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    print(f"Загружено {len(df)} записей из базы данных")
    return df

def get_available_shops_and_gaps(db_path):
    """Получение списка доступных магазинов и категорий"""
    conn = sqlite3.connect(db_path)
    shops = pd.read_sql_query("SELECT DISTINCT Магазин FROM sales_data ORDER BY Магазин", conn)['Магазин'].tolist()
    gaps = pd.read_sql_query("SELECT DISTINCT GAP FROM sales_data ORDER BY GAP", conn)['GAP'].tolist()
    conn.close()
    return shops, gaps

def create_super_features(df):
    """Создание супер-фич на основе 3 лет данных"""
    print("Создание СУПЕР-ФИЧЕЙ на основе 3-летних данных...")
    
    # 1. ГОДОВЫЕ СЕЗОННЫЕ ФИЧИ
    df['day_of_year'] = df['Дата'].dt.dayofyear
    df['week_of_year'] = df['Дата'].dt.isocalendar().week
    df['is_year_end'] = df['day_of_year'].isin([365, 366]).astype(int)
    df['is_year_start'] = (df['day_of_year'] == 1).astype(int)
    
    # 2. КВАРТАЛЬНЫЕ ФИЧИ
    df['quarter'] = df['Дата'].dt.quarter
    df['is_quarter_end'] = df['Дата'].dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df['Дата'].dt.is_quarter_start.astype(int)
    
    # 3. ПРАЗДНИЧНЫЕ КЛАСТЕРЫ
    # Группируем праздники по типам
    holiday_mapping = {
        '1': 'official_holiday',
        '0': 'no_holiday'
    }
    df['holiday_type'] = df['is_holiday_day'].map(holiday_mapping)
    
    # 4. СЛОЖНЫЕ ВЗАИМОДЕЙСТВИЯ
    df['weekend_holiday'] = ((df['is_holiday_day'] == '1') & 
                            (df['Дата'].dt.dayofweek.isin([5, 6]))).astype(int)
    df['promo_holiday'] = ((df['promoFlag'] == '1') & 
                          (df['is_holiday_day'] == '1')).astype(int)
    
    # 5. ИСТОРИЧЕСКИЕ ПАТТЕРНЫ (3 года!)
    for months in [3, 6, 12]:
        df[f'same_period_{months}m_ago'] = df.groupby(['Магазин', 'НоменклатураКод'])[
            'Количество'].transform(lambda x: x.shift(30 * months))
    
    # 6. СТАТИСТИКИ ПО ГОДАМ
    df['year'] = df['Дата'].dt.year
    for year in df['year'].unique():
        year_avg = df[df['year'] == year].groupby('НоменклатураКод')['Количество'].mean()
        df[f'avg_sales_year_{year}'] = df['НоменклатураКод'].map(year_avg)
    
    # 7. ТРЕНДОВЫЕ ФИЧИ
    df['sales_trend_7d'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
        lambda x: x.rolling(7, min_periods=1).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, 
            raw=False
        )
    )
    
    # 8. ФИЧИ ИЗМЕНЧИВОСТИ
    df['sales_volatility_14d'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
        lambda x: x.rolling(14, min_periods=1).std().fillna(0)
    )
    
    print("СУПЕР-ФИЧИ созданы!")
    return df

def create_peak_features(df):
    """Улучшенные фичи для пиков продаж"""
    print("Создание улучшенных фичей пиков продаж...")
    
    # Адаптивный порог пика для каждого товара
    df['is_peak'] = 0
    df['peak_strength'] = 0.0
    
    for (store, item), group in df.groupby(['Магазин', 'НоменклатураКод']):
        if len(group) > 7:  # Минимум неделя данных
            item_sales = group['Количество'].values
            Q3 = np.percentile(item_sales, 75)
            Q1 = np.percentile(item_sales, 25)
            iqr = Q3 - Q1
            peak_threshold = Q3 + 1.5 * iqr
            
            mask = (df['Магазин'] == store) & (df['НоменклатураКод'] == item)
            df.loc[mask, 'is_peak'] = (df.loc[mask, 'Количество'] > peak_threshold).astype(int)
            df.loc[mask, 'peak_strength'] = (df.loc[mask, 'Количество'] - peak_threshold) / (peak_threshold + 1e-8)
    
    # Скользящие паттерны пиков
    for window in [7, 14, 30]:
        df[f'peak_frequency_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['is_peak'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    print("Улучшенные фичи пиков созданы")
    return df

def create_seasonal_features(df):
    """Создание сезонных и календарных фич - УЛУЧШЕННАЯ ВЕРСИЯ"""
    print("Создание сезонных фич...")
    
    df['weekday'] = df['Дата'].dt.weekday.astype(str)
    df['day'] = df['Дата'].dt.day.astype(str)
    df['month'] = df['Дата'].dt.month.astype(str)
    df['year'] = df['Дата'].dt.year.astype(str)
    
    # БАЗОВЫЕ ФИЧИ ДНЕЙ НЕДЕЛИ
    df['is_friday'] = (df['Дата'].dt.weekday == 4).astype(int)
    df['is_saturday'] = (df['Дата'].dt.weekday == 5).astype(int)
    df['is_sunday'] = (df['Дата'].dt.weekday == 6).astype(int)
    df['is_monday'] = (df['Дата'].dt.weekday == 0).astype(int)
    
    # УЛУЧШЕННЫЕ ФИЧИ ПРАЗДНИКОВ
    df['days_before_holiday'] = 0
    df['days_after_holiday'] = 0
    
    # Находим праздничные дни и рассчитываем расстояния
    holiday_dates = df[df['is_holiday_day'] == '1']['Дата'].unique()
    
    for holiday_date in holiday_dates:
        mask_before = (df['Дата'] >= holiday_date - pd.Timedelta(days=7)) & (df['Дата'] < holiday_date)
        mask_after = (df['Дата'] > holiday_date) & (df['Дата'] <= holiday_date + pd.Timedelta(days=7))
        
        df.loc[mask_before, 'days_before_holiday'] = (holiday_date - df.loc[mask_before, 'Дата']).dt.days
        df.loc[mask_after, 'days_after_holiday'] = (df.loc[mask_after, 'Дата'] - holiday_date).dt.days
    
    # КОМБИНАЦИЯ ПРАЗДНИК + ДИСКОНТ
    df['holiday_discount'] = (df['is_holiday_day'] == '1').astype(int) * (df['promoFlag'] == '1').astype(int)
    
    # ДИСКОНТ + ВЫХОДНЫЕ
    df['discount_weekend'] = (df['promoFlag'] == '1').astype(int) * ((df['is_saturday'] == 1) | (df['is_sunday'] == 1))
    
    print("Улучшенные сезонные фичи созданы")
    return df

def create_smart_lag_features(df):
    """Улучшенные лаг-фичи для длинных горизонтов - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    print("Создание улучшенных лаг-фичей...")
    
    df.sort_values(['Магазин', 'НоменклатураКод', 'Дата'], inplace=True)
    
    # ОПТИМИЗИРОВАННЫЕ ЛАГИ для разных горизонтов
    base_lags = [1, 7, 14, 21]  # Базовые лаги для всех горизонтов
    
    # РАСШИРЕННЫЕ ЛАГИ специально для 60 дней
    extended_lags_60d = [28, 35, 42, 56, 60]  # 4, 5, 6, 8, 9 недель
    
    # Все лаги
    all_lags = base_lags + extended_lags_60d
    
    for lag in all_lags:
        df[f'lag_{lag}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].shift(lag)
        # ЛУЧШЕЕ ЗАПОЛНЕНИЕ ПРОПУСКОВ - медианой по группе
        df[f'lag_{lag}'] = df.groupby(['Магазин', 'НоменклатураКод'])[f'lag_{lag}'].transform(
            lambda x: x.fillna(x.median() if not x.isnull().all() else 0)
        )
        print(f"Создан lag_{lag}")
    
    # УЛУЧШЕННЫЕ ROLLING FEATURES
    windows = {
        'short': [3, 7],      
        'medium': [14, 21],   
        'long': [30, 45, 60]  # Добавили 45 дней для 60-дневного горизонта
    }
    
    for period_name, period_windows in windows.items():
        for window in period_windows:
            df[f'rolling_mean_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
            # Добавляем rolling min/max для обнаружения диапазонов
            df[f'rolling_min_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            print(f"Созданы rolling features для {window} дней ({period_name})")
    
    # УЛУЧШЕННЫЕ ТРЕНДОВЫЕ ФИЧИ
    for window in [7, 14, 30]:
        df[f'trend_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
            lambda x: x.rolling(window=window, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, 
                raw=False
            ).fillna(0)
        )
    
    # СЕЗОННЫЕ ФИЧИ ВЫСОКОГО ПОРЯДКА
    df['month_sin'] = np.sin(2 * np.pi * df['Дата'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Дата'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Дата'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Дата'].dt.dayofyear / 365)
    
    print("Улучшенные фичи созданы")
    return df

def prepare_retail_data_advanced(df):
    """Продвинутая подготовка розничных данных - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    
    print(f"Всего данных: {len(df)} записей")
    print(f"Уникальные товары: {df['НоменклатураКод'].nunique()}")
    print(f"Диапазон дат: {df['Дата'].min()} - {df['Дата'].max()}")
    
    # Анализ распределения целевой переменной
    target_stats = df['Количество'].describe()
    
    # ДЕТАЛЬНЫЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПРОДАЖ
    print(f"\nДетальный анализ продаж:")
    for percentile in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
        value = df['Количество'].quantile(percentile)
        print(f"{percentile*100:.1f}%: {value:.1f} единиц\n")
    
    # Анализ частоты высоких продаж
    high_sales = df[df['Количество'] > 4]
    print(f"\nПродажи > 4 единиц: {len(high_sales)} записей ({len(high_sales)/len(df)*100:.2f}%)")
    
    if len(high_sales) > 0:
        print("Примеры высоких продаж:")
        for _, row in high_sales.head(10).iterrows():
            print(f"  {row['Дата']} - {row['Магазин']} - {row['НоменклатураКод']}: {row['Количество']} ед.")
    
    df_transformed = df.copy()
    
    # УМНОЕ КЭППИНГ - СОХРАНЯЕМ РЕАЛЬНЫЕ ПРОДАЖИ
    # Вместо 95% перцентиля используем 99.5% перцентиль или разумный максимум
    reasonable_max = df_transformed['Количество'].quantile(0.995)
    
    # Если 99.5% перцентиль слишком низкий, используем эвристику
    if reasonable_max < 10:  # Если 99.5% все еще меньше 10, берем больше
        reasonable_max = max(25, df_transformed['Количество'].quantile(0.999))
    
    print(f"\nУмное кэппинг выбросов:")
    print(f"  Обрезаем только экстремальные значения выше: {reasonable_max:.1f} единиц")
    
    # Сохраняем исходные значения для анализа
    original_max = df_transformed['Количество'].max()
    clipped_count = (df_transformed['Количество'] > reasonable_max).sum()
    
    df_transformed['Количество'] = np.where(
        df_transformed['Количество'] > reasonable_max, 
        reasonable_max, 
        df_transformed['Количество']
    )
    
    print(f"  Было макс: {original_max:.1f}, стало макс: {df_transformed['Количество'].max():.1f}")
    print(f"  Обрезано записей: {clipped_count} ({clipped_count/len(df)*100:.3f}%)")
    
    # ОБРАБОТКА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ (возвратов)
    negative_count = (df_transformed['Количество'] < 0).sum()
    if negative_count > 0:
        print(f"  Обработано отрицательных значений: {negative_count}")
        df_transformed['Количество'] = np.maximum(df_transformed['Количество'], 0)
    
    # Анализ нового распределения
    new_stats = df_transformed['Количество'].describe()
    print(f"\nНовое распределение:")
    print(f"  75%: {new_stats['75%']:.1f}, 90%: {df_transformed['Количество'].quantile(0.90):.1f}")
    print(f"  95%: {df_transformed['Количество'].quantile(0.95):.1f}")
    print(f"  99%: {df_transformed['Количество'].quantile(0.99):.1f}")
    
    df_transformed = df_transformed.sort_values(['Магазин', 'НоменклатураКод', 'Дата'])
    df_transformed['days_since_last_sale'] = df_transformed.groupby(
        ['Магазин', 'НоменклатураКод']
    )['Дата'].diff().dt.days.fillna(0)
    
    # Бинарная фича - были ли продажи в последние N дней
    for days in [7, 14, 30]:
        df_transformed[f'had_sales_last_{days}d'] = (
            df_transformed['days_since_last_sale'] <= days
        ).astype(int)
    
    # ДОБАВЛЯЕМ ФИЧИ ПИКОВ 
    df_transformed = create_peak_features(df_transformed)
    
    return df_transformed

def prepare_data_from_db(db_path, selected_shops=None, selected_gaps=None):
    """Подготовка данных"""
    print("Загрузка данных из базы данных...")
    
    df = load_data_from_db(db_path, shops=selected_shops, gaps=selected_gaps)

    if len(df) == 0:
        raise ValueError("Не удалось загрузить данные из базы данных")
    
    # Удаляем проблемные столбцы
    columns_to_drop = ['Вид', 'Подкатегория', 'Номенклатура', 'Страна']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        print(f"Удалены столбцы: {existing_columns_to_drop}")
    
    # Базовые преобразования
    df['Дата'] = pd.to_datetime(df['Дата'])
    df = prepare_retail_data_advanced(df)
    
    print(f"\nИсходное распределение целевой переменной:")
    print(f"  Нулевых значений: {(df['Количество'] == 0).sum()} ({(df['Количество'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Ненулевых значений: {(df['Количество'] > 0).sum()} ({(df['Количество'] > 0).sum()/len(df)*100:.1f}%)")
    
    # Создание фич
    df = create_seasonal_features(df)
    aggressive_memory_cleanup()
    df = create_smart_lag_features(df)
    aggressive_memory_cleanup()
    
    # Преобразование в категории
    categorical_columns = [
        'Магазин', 'НоменклатураКод', 'GAP',
        'promoFlag', 'is_holiday_day', 'weekday', 'day', 'month', 'year',
        'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
        'is_end_of_month', 'is_beginning_of_month', 'is_salary_week'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category')
    
    # Временные индексы
    min_date = df["Дата"].min()
    df["time_idx"] = (df["Дата"] - min_date).dt.days
    
    df["ГОД"] = df["Дата"].dt.year.astype(str).astype('category')
    df["Месяц"] = df["Дата"].dt.month.astype(str).astype('category')
    
    print(f"Данные подготовлены: {len(df)} записей")
    return df
    

def aggregate_daily_data(df):
    """Правильная агрегация - только существующие комбинации"""
    print("Агрегация данных по дням")
    print(f"До агрегации: {len(df)} записей")
    
    group_cols = ['Магазин', 'НоменклатураКод', 'time_idx', 'Дата']
    # Проверяем, что все колонки есть
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        print(f"Отсутствуют колонки: {missing_cols}")
        return df
    
    # ПРОСТАЯ агрегация - сумма продаж по дням
    df_agg = df.groupby(group_cols)['Количество'].sum().reset_index()
    
    print(f"После агрегации: {len(df_agg)} записей")
    
    if len(df_agg) > len(df) * 2:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Слишком много записей после агрегации!")
        print("Используем исходные данные...")
        return df
    
    return df_agg


# def handle_sparse_data_simple(df_train, df_val, df_test):
#     """Простая обработка спарс данных"""
    
#     item_sales = df_train.groupby('НоменклатураКод')['Количество'].sum()
#     active_items = item_sales[item_sales > 0].index.tolist()
    
#     print(f"Активных товаров (с продажами > 0): {len(active_items)} из {len(item_sales)}")
    
#     df_train_filtered = df_train[df_train['НоменклатураКод'].isin(active_items)].copy()
#     df_val_filtered = df_val[df_val['НоменклатураКод'].isin(active_items)].copy() if len(df_val) > 0 else df_val
#     df_test_filtered = df_test[df_test['НоменклатураКод'].isin(active_items)].copy() if len(df_test) > 0 else df_test
    
#     print(f"После фильтрации неактивных товаров:")
#     print(f"  Train: {len(df_train_filtered)} записей (было {len(df_train)})")
#     print(f"  Val: {len(df_val_filtered)} записей (было {len(df_val)})") 
#     print(f"  Test: {len(df_test_filtered)} записей (было {len(df_test)})")
    
#     for name, df in [("Train", df_train_filtered), ("Val", df_val_filtered), ("Test", df_test_filtered)]:
#         if len(df) == 0:
#             continue
#         target = df['Количество'].values
#         print(f"{name} после фильтрации:")
#         print(f"  Нулевых значений: {(target == 0).sum()} ({(target == 0).sum()/len(target)*100:.1f}%)")
#         print(f"  Среднее количество: {target.mean():.4f}")
    
#     return df_train_filtered, df_val_filtered, df_test_filtered

def split_by_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.2):
    """Разделение данных - СОХРАНЯЕМ ВСЕ МАГАЗИНЫ"""
    print("Умное разделение данных с сохранением всех магазинов...")
    
    # НЕ УБИРАЕМ магазины с малым количеством данных!
    # Вместо этого используем стратегию для каждого магазина отдельно
    
    result_train = []
    result_val = []
    # result_test = []
    
    for shop in df['Магазин'].unique():
        shop_data = df[df['Магазин'] == shop].copy()
        shop_data = shop_data.sort_values("time_idx")
        
        unique_times = sorted(shop_data["time_idx"].unique())
        
        # Если данных мало для разделения, используем только train
        if len(unique_times) < 50:
            print(f"  {shop}: мало данных ({len(unique_times)} дней) - только train")
            result_train.append(shop_data)
            continue
            
        train_cutoff = int(train_ratio * len(unique_times))
        val_cutoff = int((train_ratio + val_ratio) * len(unique_times))
        
        train_times = unique_times[:train_cutoff]
        val_times = unique_times[train_cutoff:val_cutoff] if val_cutoff > train_cutoff else []
        test_times = unique_times[val_cutoff:] if len(unique_times) > val_cutoff else []
        
        shop_train = shop_data[shop_data["time_idx"].isin(train_times)].copy()
        shop_val = shop_data[shop_data["time_idx"].isin(val_times)].copy() if val_times else pd.DataFrame()
        shop_test = shop_data[shop_data["time_idx"].isin(test_times)].copy() if test_times else pd.DataFrame()
        
        result_train.append(shop_train)
        if len(shop_val) > 0:
            result_val.append(shop_val)
        # if len(shop_test) > 0:
        #     result_test.append(shop_test)
            
        print(f"  {shop}: train={len(shop_train)}, val={len(shop_val)}")
    
    # Собираем результаты
    df_train = pd.concat(result_train, ignore_index=True) if result_train else pd.DataFrame()
    df_val = pd.concat(result_val, ignore_index=True) if result_val else pd.DataFrame()
    # df_test = pd.concat(result_test, ignore_index=True) if result_test else pd.DataFrame()
    
    print(f"Итоговое разделение:")
    print(f"  Train: {len(df_train)} записей, {df_train['Магазин'].nunique()} магазинов")
    print(f"  Val: {len(df_val)} записей, {df_val['Магазин'].nunique()} магазинов")
    # print(f"  Test: {len(df_test)} записей, {df_test['Магазин'].nunique()} магазинов")
    
    # Проверяем, что все магазины есть хотя бы в train
    all_shops = set(df['Магазин'].unique())
    train_shops = set(df_train['Магазин'].unique())
    missing_shops = all_shops - train_shops
    
    if missing_shops:
        print(f"Внимание: эти магазины отсутствуют даже в train: {missing_shops}")
        # Добавляем недостающие магазины в train
        for shop in missing_shops:
            shop_data = df[df['Магазин'] == shop].copy()
            result_train.append(shop_data)
            print(f"  Добавили {shop} в train: {len(shop_data)} записей")
        
        df_train = pd.concat(result_train, ignore_index=True)
    
    return df_train, df_val, pd.DataFrame()

def check_shop_data_distribution(df):
    """Проверка распределения данных по магазинам"""
    print("\n   РАСПРЕДЕЛЕНИЕ ДАННЫХ ПО МАГАЗИНАМ")
    
    shop_stats = df.groupby('Магазин').agg({
        'НоменклатураКод': 'nunique',
        'Дата': ['min', 'max', 'nunique'],
        'time_idx': ['min', 'max']
    }).reset_index()
    
    shop_stats.columns = ['Магазин', 'уникальных_товаров', 'первая_дата', 'последняя_дата', 'уникальных_дат', 'min_time_idx', 'max_time_idx']
    
    shop_stats['дней_данных'] = (shop_stats['последняя_дата'] - shop_stats['первая_дата']).dt.days + 1
    shop_stats['записей'] = df.groupby('Магазин').size().values
    
    print(shop_stats.to_string(index=False))
    return shop_stats

def scale_numeric(train_df, val_df, test_df):
    """Масштабирование с MinMaxScaler"""
    target_col = 'Количество'
    if target_col not in train_df.columns:
        print("Предупреждение: нет столбца 'Количество' для масштабирования")
        return train_df, val_df, test_df, None
    
    print("Статистика перед масштабированием:")
    print(f"  Train - min: {train_df[target_col].min():.4f}, max: {train_df[target_col].max():.4f}")
    
    # Используем MinMaxScaler вместо StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    # Обрезаем отрицательные значения
    train_df[target_col] = np.maximum(train_df[target_col], 0)
    if len(val_df) > 0:
        val_df[target_col] = np.maximum(val_df[target_col], 0)
    # if len(test_df) > 0:
    #     test_df[target_col] = np.maximum(test_df[target_col], 0)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_df[target_col] = scaler.fit_transform(train_df[[target_col]])
    
    if len(val_df) > 0:
        val_df[target_col] = scaler.transform(val_df[[target_col]])
    
    # if len(test_df) > 0:
    #     test_df[target_col] = scaler.transform(test_df[[target_col]])
    
    print("Статистика после MinMaxScaler:")
    print(f"  Train - min: {train_df[target_col].min():.4f}, max: {train_df[target_col].max():.4f}")
    
    return train_df, val_df, test_df, scaler

# -----------------------------------------------------------------------------
# DATASET И DATALOADER
# -----------------------------------------------------------------------------
def make_dataloader(ts_ds: TimeSeriesDataSet, train: bool, batch_size: int):
    """Создание DataLoader для датасета"""
    return ts_ds.to_dataloader(
        train=train,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        drop_last=False,
        shuffle=train,
    )

def create_simple_model(train_ds, horizon_params):
    """Создание простой модели TFT"""
    try:
        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=horizon_params["learning_rate"],
            hidden_size=horizon_params["hidden_size"],
            lstm_layers=horizon_params["lstm_layers"],
            attention_head_size=horizon_params["attention_head_size"],
            dropout=horizon_params["dropout"],
            hidden_continuous_size=horizon_params["hidden_continuous_size"],
            output_size=1,
            loss=SMAPE(),
            reduce_on_plateau_patience=3,
        )
        return model
    except Exception as e:
        print(f"Ошибка создания модели: {e}")
        return None
    

def create_simple_dataset(df, prediction_length):
    """Создание датасета с исправлением категорий"""
    print("   СОЗДАНИЕ УПРОЩЕННОГО ДАТАСЕТА")
    
    # Копируем данные
    df_simple = df[['Магазин', 'НоменклатураКод', 'time_idx', 'Количество', 'weekday', 'promoFlag', 'is_holiday_day']].copy()
    
    # Убедимся, что все категории согласованы
    categorical_cols = ['Магазин', 'НоменклатураКод', 'weekday', 'promoFlag', 'is_holiday_day']
    for col in categorical_cols:
        if col in df_simple.columns:
            df_simple[col] = df_simple[col].astype(str).astype('category')
    
    dataset = TimeSeriesDataSet(
        df_simple,
        time_idx="time_idx",
        target="Количество",
        group_ids=["Магазин", "НоменклатураКод"],
        min_encoder_length=10,
        max_encoder_length=30,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=["Магазин", "НоменклатураКод"],
        time_varying_known_categoricals=["weekday", "promoFlag", "is_holiday_day"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["Количество"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        categorical_encoders={
            "Магазин": NaNLabelEncoder(add_nan=True),
            "НоменклатураКод": NaNLabelEncoder(add_nan=True),
            "weekday": NaNLabelEncoder(add_nan=True),
            "promoFlag": NaNLabelEncoder(add_nan=True),
            "is_holiday_day": NaNLabelEncoder(add_nan=True)
        }
    )
    
    return dataset

def create_simple_60_days_dataset(df, prediction_length):
    """Упрощенный датасет для 60 дней"""
    print("Пробуем упрощенный подход...")
    
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Количество",
        group_ids=["Магазин"],
        min_encoder_length=15,
        max_encoder_length=30,
        min_prediction_length=prediction_length, 
        max_prediction_length=prediction_length,
        static_categoricals=["Магазин"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["Количество"],
        add_relative_time_idx=False,  # Отключаем сложные фичи
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=True,
    )
    
    print(f"Упрощенный датасет создан: {len(dataset)} рядов")
    return dataset, prediction_length

def create_super_simple_60_days_dataset(df, prediction_length):
    """СУПЕР-УПРОЩЕННЫЙ датасет для 60 дней"""
    print("=== СОЗДАНИЕ СУПЕР-УПРОЩЕННОГО ДАТАСЕТА ДЛЯ 60 ДНЕЙ ===")
    
    # Используем только самые базовые фичи
    basic_features = ['Магазин', 'НоменклатураКод', 'time_idx', 'Количество']
    
    try:
        dataset = TimeSeriesDataSet(
            df[basic_features],
            time_idx="time_idx",
            target="Количество", 
            group_ids=["Магазин", "НоменклатураКод"],
            min_encoder_length=15,  # Еще меньше
            max_encoder_length=30,  # Еще меньше
            min_prediction_length=prediction_length,
            max_prediction_length=prediction_length,
            static_categoricals=["Магазин", "НоменклатураКод"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["Количество"],
            add_relative_time_idx=False,
            add_target_scales=False,  # Отключаем для простоты
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={
                "Магазин": NaNLabelEncoder(add_nan=True),
                "НоменклатураКод": NaNLabelEncoder(add_nan=True)
            }
        )
        
        print(f"Супер-упрощенный датасет создан: {len(dataset)} рядов")
        return dataset, prediction_length
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None, prediction_length

# def create_dataset_for_60_days(df, prediction_length):
#     """Создание датасета для 60 дней - ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ"""
#     print("   СОЗДАНИЕ ДАТАСЕТА ДЛЯ 60 ДНЕЙ (ФИКС)")
    
#     # ФИЛЬТРАЦИЯ: оставляем только ряды достаточной длины
#     series_lengths = df.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
#         lambda x: x.max() - x.min() + 1
#     )
    
#     # Минимальная длина для 60 дней: encoder(30) + prediction(60) + buffer(5) = 95 дней
#     min_required_length = 95
#     valid_series = series_lengths[series_lengths >= min_required_length].index.tolist()
    
#     print(f"До фильтрации: {len(series_lengths)} рядов")
#     print(f"После фильтрации (>={min_required_length} дней): {len(valid_series)} рядов")
    
#     if len(valid_series) < 50:
#         print(f"Слишком мало рядов после фильтрации: {len(valid_series)}")
#         return None, prediction_length
    
#     # Фильтруем данные
#     df_filtered = df.set_index(['Магазин', 'НоменклатураКод']).loc[valid_series].reset_index()
    
#     # СУПЕР-УПРОЩЕННЫЙ ДАТАСЕТ
#     basic_features = ['Магазин', 'НоменклатураКод', 'time_idx', 'Количество']
    
#     try:
#         dataset = TimeSeriesDataSet(
#             df_filtered[basic_features],
#             time_idx="time_idx",
#             target="Количество",
#             group_ids=["Магазин", "НоменклатураКод"],
#             min_encoder_length=30,  # Фиксированный энкодер
#             max_encoder_length=30,  # Фиксированный энкодер
#             min_prediction_length=prediction_length,
#             max_prediction_length=prediction_length,
#             static_categoricals=["Магазин", "НоменклатураКод"],
#             time_varying_known_reals=["time_idx"],
#             time_varying_unknown_reals=["Количество"],
#             add_relative_time_idx=False,  # Отключаем сложные фичи
#             add_target_scales=True,
#             add_encoder_length=True,
#             allow_missing_timesteps=True,
#             categorical_encoders={
#                 "Магазин": NaNLabelEncoder(add_nan=True),
#                 "НоменклатураКод": NaNLabelEncoder(add_nan=True)
#             }
#         )
        
#         print(f"Датсет для 60 дней создан: {len(dataset)} временных рядов")
#         return dataset, prediction_length
        
#     except Exception as e:
#         print(f"Критическая ошибка создания датасета: {e}")
#         return None, prediction_length

def create_smart_60d_dataset_all_data(df_train, df_val, df_test, prediction_length):
    """Умное создание датасета с использованием ВСЕХ данных"""
    print("   УМНОЕ СОЗДАНИЕ ДАТАСЕТА (ВСЕ ДАННЫЕ)")
    
    # Анализ длин рядов в тренировочных данных
    train_series_lengths = df_train.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
        lambda x: x.max() - x.min() + 1
    )
    
    print(f"Тренировочные данные: {len(train_series_lengths)} рядов")
    print(f"  Мин: {train_series_lengths.min()} дней")
    print(f"  Макс: {train_series_lengths.max()} дней")
    
    # РАЗДЕЛЯЕМ РЯДЫ ПО ДЛИНЕ ДЛЯ РАЗНЫХ СТРАТЕГИЙ
    very_short = train_series_lengths[train_series_lengths < 30].index.tolist()      # <30 дней - проблемные
    short = train_series_lengths[(train_series_lengths >= 30) & (train_series_lengths < 95)].index.tolist()  # 30-94 дней
    long = train_series_lengths[train_series_lengths >= 95].index.tolist()           # >=95 дней - идеальные
    
    print(f"Классификация рядов:")
    print(f"  Очень короткие (<30 дней): {len(very_short)} рядов")
    print(f"  Короткие (30-94 дней): {len(short)} рядов") 
    print(f"  Длинные (>=95 дней): {len(long)} рядов")
    
    # СТРАТЕГИЯ: используем ВСЕ данные, но с разными параметрами
    try:
        # ПАРАМЕТРЫ ДЛЯ КОРОТКИХ РЯДОВ
        min_encoder_short = 10  # Минимум для коротких рядов
        max_encoder_short = 20  # Максимум для коротких рядов
        
        # ПАРАМЕТРЫ ДЛЯ ДЛИННЫХ РЯДОВ  
        min_encoder_long = 30
        max_encoder_long = 30
        
        print(f"  Параметры энкодера:")
        print(f"  Короткие ряды: {min_encoder_short}-{max_encoder_short} дней")
        print(f"  Длинные ряды: {min_encoder_long}-{max_encoder_long} дней")
        
        # СОЗДАЕМ ДАТАСЕТ С ГИБКИМИ ПАРАМЕТРАМИ
        train_ds = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="Количество",
            group_ids=["Магазин", "НоменклатураКод"],
            min_encoder_length=min_encoder_short,  # Меньше для коротких рядов
            max_encoder_length=max_encoder_long,   # Больше для длинных рядов
            min_prediction_length=prediction_length,
            max_prediction_length=prediction_length,
            static_categoricals=["Магазин", "НоменклатураКод"],
            time_varying_known_categoricals=["weekday", "promoFlag", "is_holiday_day"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["Количество"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={
                "Магазин": NaNLabelEncoder(add_nan=True),
                "НоменклатураКод": NaNLabelEncoder(add_nan=True)
            }
        )
        
        print(f"Датсет создан: {len(train_ds)} рядов")
        print(f"Использовано {len(train_series_lengths)} из {len(train_series_lengths)} рядов (100%)")
        
        # Валидация и тест (только длинные ряды)
        val_ds = None
        if len(df_val) > 0:
            # Фильтруем валидацию - только длинные ряды
            val_series_lengths = df_val.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
                lambda x: x.max() - x.min() + 1
            )
            long_val_series = val_series_lengths[val_series_lengths >= 95].index.tolist()
            df_val_filtered = df_val.set_index(['Магазин', 'НоменклатураКод']).loc[long_val_series].reset_index()
            
            if len(df_val_filtered) > 0:
                val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val_filtered, predict=True)
                print(f"Валидационный датасет: {len(df_val_filtered)} записей ({len(long_val_series)} рядов)")
            else:
                print("Нет длинных рядов для валидации")
        
        test_ds = None
        if len(df_test) > 0:
            # Фильтруем тест - только длинные ряды
            test_series_lengths = df_test.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
                lambda x: x.max() - x.min() + 1
            )
            long_test_series = test_series_lengths[test_series_lengths >= 95].index.tolist()
            df_test_filtered = df_test.set_index(['Магазин', 'НоменклатураКод']).loc[long_test_series].reset_index()
            
            if len(df_test_filtered) > 0:
                test_ds = TimeSeriesDataSet.from_dataset(train_ds, df_test_filtered, predict=True)
                print(f"Тестовый датасет: {len(df_test_filtered)} записей ({len(long_test_series)} рядов)")
            else:
                print("Нет длинных рядов для теста")
        
        return train_ds, val_ds, test_ds, prediction_length
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None, None, prediction_length

def check_60_days_viability(df):
    """Проверка возможности обучения на 60 дней"""
    print("ПРОВЕРКА ВОЗМОЖНОСТИ ОБУЧЕНИЯ НА 60 ДНЕЙ")
    
    # Анализ длины рядов
    series_lengths = df.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
        lambda x: x.max() - x.min() + 1
    )
    
    # Требования для 60 дней
    required_length = 30 + 60 + 5  # encoder + prediction + buffer
    sufficient_series = series_lengths[series_lengths >= required_length]
    
    print(f"Рядов достаточно для 60 дней: {len(sufficient_series)}/{len(series_lengths)}")
    print(f"Минимальная требуемая длина: {required_length} дней")
    print(f"Фактическая минимальная длина: {series_lengths.min()} дней")
    print(f"Фактическая максимальная длина: {series_lengths.max()} дней")
    
    # Анализ распределения длин
    length_stats = series_lengths.describe()
    print(f"Распределение длин рядов:")
    print(f"  25%: {length_stats['25%']} дней")
    print(f"  50%: {length_stats['50%']} дней") 
    print(f"  75%: {length_stats['75%']} дней")
    
    if len(sufficient_series) < 100:
        print("ВНИМАНИЕ: Слишком мало рядов для 60-дневного горизонта")
        print("Рекомендуется использовать меньший горизонт или собрать больше данных")
        return False
    
    return True

def analyze_series_lengths_detailed(df):
    """Детальный анализ длин рядов"""
    print("\n=== ДЕТАЛЬНЫЙ АНАЛИЗ ДЛИН РЯДОВ ===")
    
    # Длины рядов для каждого товара в каждом магазине
    series_lengths = df.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
        lambda x: x.max() - x.min() + 1
    )
    
    # Статистика
    print("ОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего рядов: {len(series_lengths)}")
    print(f"  Мин: {series_lengths.min()} дней")
    print(f"  Макс: {series_lengths.max()} дней")
    print(f"  Среднее: {series_lengths.mean():.1f} дней")
    print(f"  Медиана: {series_lengths.median()} дней")
    
    # Распределение по длинам
    length_bins = [1, 7, 30, 90, 180, 365, float('inf')]
    bin_labels = ['1 день', '2-7 дней', '8-30 дней', '31-90 дней', '91-180 дней', '181-365 дней', '>365 дней']
    
    print("\nРАСПРЕДЕЛЕНИЕ ПО ДЛИНАМ:")
    for i in range(len(length_bins)-1):
        if i == 0:
            count = (series_lengths == 1).sum()
            label = '1 день'
        else:
            count = ((series_lengths > length_bins[i]) & (series_lengths <= length_bins[i+1])).sum()
            label = f"{length_bins[i]+1}-{length_bins[i+1]} дней"
        
        percentage = count / len(series_lengths) * 100
        print(f"  {label}: {count} рядов ({percentage:.1f}%)")
    
    # Примеры самых коротких рядов
    shortest_series = series_lengths.nsmallest(10)
    print(f"\nПРИМЕРЫ САМЫХ КОРОТКИХ РЯДОВ (1 день):")
    for (shop, item), length in shortest_series[shortest_series == 1].items():
        shop_data = df[(df['Магазин'] == shop) & (df['НоменклатураКод'] == item)]
        dates = shop_data['Дата'].unique()
        print(f"  {shop} - {item}: {length} день, дата: {dates[0]}")
    
    # Анализ для 60 дней
    required_for_60d = 95  # encoder(30) + prediction(60) + buffer(5)
    sufficient_for_60d = series_lengths[series_lengths >= required_for_60d]
    
    print(f"\nАНАЛИЗ ДЛЯ 60 ДНЕЙ:")
    print(f"  Требуется: {required_for_60d}+ дней")
    print(f"  Достаточно длинные ряды: {len(sufficient_for_60d)}/{len(series_lengths)} ({len(sufficient_for_60d)/len(series_lengths)*100:.1f}%)")
    
    if len(sufficient_for_60d) > 0:
        print(f"  Примеры достаточно длинных рядов:")
        for (shop, item), length in sufficient_for_60d.head(5).items():
            print(f"    {shop} - {item}: {length} дней")
    
    return series_lengths

def fine_tune_60d_model(df_train, df_val, df_test):
    """Простой файн-тюнинг модели 30 дней для 60 дней"""
    print("    ПРОСТОЙ ФАЙН-ТЮНИНГ ДЛЯ 60 ДНЕЙ")
    
    try:
        # Загружаем модель для 30 дней
        model_30d_path = os.path.join(MODEL_DIR, "tft_30d.pth")
        if not os.path.exists(model_30d_path):
            print("Модель для 30 дней не найдена")
            return None
        
        # Создаем упрощенный датасет для 60 дней
        train_ds_60d = create_super_simple_60_days_dataset(df_train, 60)
        if train_ds_60d is None:
            return None
        
        # Создаем новую модель с теми же параметрами, но для 60 дней
        horizon_params = get_adaptive_hyperparameters(df_train, 60)
        
        from pytorch_forecasting.metrics import MAE
        
        model_60d = TemporalFusionTransformer.from_dataset(
            train_ds_60d,
            learning_rate=horizon_params["learning_rate"] * 0.5,  # Половина LR
            hidden_size=horizon_params["hidden_size"],
            lstm_layers=1,
            attention_head_size=1,
            dropout=0.2,
            hidden_continuous_size=8,
            output_size=1,
            loss=MAE(),
            reduce_on_plateau_patience=5,
        )
        
        print("Модель для 60 дней создана, начинаем обучение...")
        
        # Обучаем с нуля (не файн-тюнинг из-за разных архитектур)
        val_ds_60d = TimeSeriesDataSet.from_dataset(train_ds_60d, df_val, predict=True) if len(df_val) > 0 else None
        
        model, train_dl, val_dl, test_dl = train_one_horizon(
            train_ds_60d, val_ds_60d, None, 
            prediction_length=60, 
            horizon_params=horizon_params
        )
        
        return model
        
    except Exception as e:
        print(f"Ошибка файн-тюнинга для 60 дней: {e}")
        return None


def check_dataset_viability(df, prediction_length):
    """Проверка возможности создания датасета"""
    shop_lengths = df.groupby('Магазин').size()
    min_length = shop_lengths.min()
    
    required_length = 30 + prediction_length + 10  # encoder + prediction + buffer
    
    print(f"Проверка датасета: min_series={min_length}, required={required_length}")
    
    if min_length < required_length:
        print(f"Внимание: короткие ряды. min={min_length}, нужно={required_length}")
        return False
    return True
        

# ТЕСТИРОВАНИЕ ДАННЫХ ДЛЯ 60 ДНЕЙ--------------------------------------------
def analyze_training_capability(df):
    """Анализ возможности обучения для 60 дней"""
    print("\n" + "="*80)
    print("АНАЛИЗ ВОЗМОЖНОСТИ ОБУЧЕНИЯ НА 60 ДНЕЙ")
    print("="*80)
    
    # Анализ данных по магазинам
    shop_analysis = df.groupby('Магазин').agg({
        'time_idx': ['min', 'max', 'nunique'],
        'НоменклатураКод': 'nunique',
        'Дата': ['min', 'max']
    }).round(2)
    
    shop_analysis.columns = ['min_time', 'max_time', 'уникальных_дней', 'уникальных_товаров', 'первая_дата', 'последняя_дата']
    shop_analysis['длина_ряда'] = shop_analysis['max_time'] - shop_analysis['min_time'] + 1
    
    print("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАННЫХ:")
    print(shop_analysis[['уникальных_дней', 'длина_ряда', 'уникальных_товаров', 'первая_дата', 'последняя_дата']].to_string())
    
    # Анализ возможности обучения для 60 дней
    horizon = 60
    min_encoder = 30  # Минимальный энкодер для качественного обучения
    
    print(f"\nАНАЛИЗ ВОЗМОЖНОСТИ ОБУЧЕНИЯ НА {horizon} ДНЕЙ:")
    print("Магазин".ljust(30) + " | Доступно дней | Требуется | Статус")
    print("-" * 70)
    
    for shop in shop_analysis.index:
        available_days = shop_analysis.loc[shop, 'уникальных_дней']
        required_days = min_encoder + horizon
        can_train = available_days >= required_days
        
        status = "ДОСТАТОЧНО" if can_train else "НЕДОСТАТОЧНО"
        print(f"{shop.ljust(30)} | {available_days:12d} | {required_days:9d} | {status}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    shops = shop_analysis.index
    available_days = shop_analysis['уникальных_дней']
    
    bars = plt.bar(shops, available_days, color=['#2E8B57', '#CD5C5C'], alpha=0.7)
    plt.axhline(y=min_encoder+horizon, color='red', linestyle='--', linewidth=2, 
                label=f'Требуется для {horizon} дней ({min_encoder+horizon}д)')
    
    plt.title('ДОСТУПНЫЕ ДНИ ДАННЫХ ДЛЯ ОБУЧЕНИЯ НА 60 ДНЕЙ', fontsize=14, fontweight='bold')
    plt.ylabel('Количество дней')
    plt.legend()
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, available_days):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{int(value)}д', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_capability_60_days.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Рекомендации
    print(f"\nРЕКОМЕНДАЦИИ:")
    for shop in shop_analysis.index:
        available_days = shop_analysis.loc


def analyze_data_quality(df, prediction_length):
    """Упрощенный анализ качества данных - только информационный"""
    print("\n=== ИНФОРМАЦИЯ О ДАННЫХ ===")
    
    # ПРАВИЛЬНЫЙ расчет длины рядов
    series_lengths = df.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
        lambda x: x.max() - x.min() + 1
    )
    
    print(f"Статистика длины рядов:")
    print(f"  Мин: {series_lengths.min()} дней")
    print(f"  Макс: {series_lengths.max()} дней") 
    print(f"  Среднее: {series_lengths.mean():.1f} дней")
    print(f"  Медиана: {series_lengths.median()} дней")
    
    # Требования для горизонтов
    if prediction_length == 30:
        required = 60 + 30 + 15
    else:  # 60 дней
        required = 90 + 60 + 15
        
    sufficient_length = series_lengths[series_lengths >= required]
    print(f"Рядов достаточно для {prediction_length} дней: {len(sufficient_length)}/{len(series_lengths)}")
    
    return True  # Всегда возвращаем True, так как горизонты жестко заданы
# -----------------------------------------------------------------------------------------------

def train_one_horizon(train_ds, val_ds, test_ds, prediction_length: int, horizon_params: dict):
    """Улучшенное обучение с автоматическими настройками - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    try:
        if train_ds is None or len(train_ds) == 0:
            logging.error("Пустой тренировочный датасет")
            return None, None, None
        
        # ПРИНУДИТЕЛЬНАЯ ПРОВЕРКА ДЛЯ 60 ДНЕЙ
        if prediction_length == 60:
            print("    СПЕЦИАЛЬНАЯ ПРОВЕРКА ДЛЯ 60 ДНЕЙ")
            if len(train_ds) < 50:
                print(f"Слишком мало рядов для 60 дней: {len(train_ds)}")
                # Уменьшаем сложность модели
                horizon_params["hidden_size"] = max(16, horizon_params["hidden_size"] // 2)
                horizon_params["lstm_layers"] = 1
                print(f"Упрощенные параметры: hidden={horizon_params['hidden_size']}")

        train_dl = make_dataloader(train_ds, train=True, batch_size=horizon_params["batch_size"])
        
        val_dl = None
        if val_ds is not None and len(val_ds) > 0:
            val_dl = make_dataloader(val_ds, train=False, batch_size=horizon_params["batch_size"] * 2)
        
        test_dl = None
        # if test_ds is not None and len(test_ds) > 0:
        #     test_dl = make_dataloader(test_ds, train=False, batch_size=horizon_params["batch_size"] * 2)

        print(f"--- Улучшенные параметры для горизонта {prediction_length} дней:")
        print(f"   Batch size: {horizon_params['batch_size']}")
        print(f"   Learning rate: {horizon_params['learning_rate']}")
        print(f"   Max epochs: {horizon_params['max_epochs']}")
        print(f"   Patience: {horizon_params.get('patience', 5)}")

        model = create_adaptive_model(train_ds, horizon_params)
        if model is None:
            return None, None, None

        # УМНЫЕ КОЛБЭКИ
        callbacks = []
        
        # Early Stopping с мониторингом валидации
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=horizon_params.get("patience", 7),
            verbose=True,
            mode="min",
            check_finite=True
        )
        callbacks.append(early_stop_callback)
        
        # Мониторинг LR
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        # Сохранение лучшей модели
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=MODEL_DIR,
            filename=f"best_{prediction_length}d_" + "{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        callbacks.append(checkpoint_callback)

        # ТРЕЙНЕР
        trainer = Trainer(
            max_epochs=horizon_params["max_epochs"],
            limit_train_batches=1.0,  
            limit_val_batches=1.0,
            accelerator="cpu",
            devices=1,
            callbacks=callbacks,
            precision=precision,
            accumulate_grad_batches=horizon_params.get("accumulate_grad_batches", 1),
            gradient_clip_val=horizon_params.get("gradient_clip_val", 0.1),
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            enable_model_summary=True,
        )

        print("=== Начинаю улучшенное обучение...")
        print_memory_usage()
        
        t0 = time.time()
        
        try:
            # Обучение с валидацией если есть данные
            if val_dl is not None:
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            else:
                # Без валидации - просто обучаем
                trainer.fit(model, train_dataloaders=train_dl)
                
            # ЗАГРУЗКА И ПЕРЕМЕЩЕНИЕ ЛУЧШЕЙ МОДЕЛИ НА ПРАВИЛЬНОЕ УСТРОЙСТВО
            if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
                print(f"Загружаем лучшую модель: {checkpoint_callback.best_model_path}")
                
                # Определяем устройство
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Загружаем модель с указанием устройства
                model = TemporalFusionTransformer.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    map_location=device
                )
                
                # Явно перемещаем модель на устройство
                model.to(device)
                print(f"Модель перемещена на устройство: {device}")
                
        except Exception as e:
            logging.error(f"Ошибка обучения: {e}")
            print(f"Детали ошибки: {type(e).__name__}: {e}")
            return None, None, None
        
        training_time = (time.time() - t0) / 60
        logging.info(f"Время обучения (h{prediction_length}): {training_time:.1f} минут")
        print(f"Обучение завершено за {training_time:.1f} минут")

        return model, train_dl, val_dl
        
    except Exception as e:
        error_logger.error(f"Ошибка в train_one_horizon: {e}")
        print(f"Критическая ошибка в train_one_horizon: {type(e).__name__}: {e}")
        return None, None, None


def postprocess_predictions(predictions, y_true=None, min_sales=0.5):
    """Умная пост-обработка прогнозов - БЕЗОПАСНАЯ ВЕРСИЯ"""
    
    # 1. Обрезаем отрицательные значения
    predictions = np.maximum(predictions, min_sales)
    
    # 2. Для дискретных данных округляем до ближайшего целого
    predictions = np.round(predictions)
    
    # 3. Ограничиваем диапазоном [1, 4] как в исходных данных
    predictions = np.clip(predictions, 1, 4)
    
    # 4. Убедимся, что нет NaN или inf
    predictions = np.nan_to_num(predictions, nan=min_sales, posinf=4.0, neginf=min_sales)
    
    return predictions


def evaluate_metrics(model, dataloader, scaler=None, max_batches=5):
    """Улучшенная оценка с анализом дискретных данных"""
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        all_preds = []
        all_targets = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"Модель на устройстве: {next(model.parameters()).device}")
        
        with torch.no_grad():
            pbar = tqdm(total=min(max_batches, len(dataloader)), 
                       desc="Улучшенная оценка")
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # Перемещаем батч на устройство
                    if isinstance(batch, (list, tuple)):
                        x, y = batch
                        x_device = {}
                        for key, value in x.items():
                            if torch.is_tensor(value):
                                x_device[key] = value.to(device)
                            else:
                                x_device[key] = value
                        
                        y_device = []
                        for item in y:
                            if torch.is_tensor(item):
                                y_device.append(item.to(device))
                            else:
                                y_device.append(item)
                        
                        batch_device = (x_device, y_device)
                    else:
                        batch_device = batch.to(device) if torch.is_tensor(batch) else batch
                    
                    # Получаем предсказания
                    if isinstance(batch_device, (list, tuple)) and len(batch_device) >= 2:
                        x, y = batch_device
                        output = model(x)
                        pred = output.prediction if hasattr(output, 'prediction') else output
                        targets = y[0] if isinstance(y, (list, tuple)) and len(y) > 0 else y
                    else:
                        output = model(batch_device)
                        pred = output.prediction if hasattr(output, 'prediction') else output
                        targets = batch_device[1] if isinstance(batch_device, (list, tuple)) and len(batch_device) > 1 else None
                    
                    if targets is None:
                        continue
                    
                    # Преобразование на CPU
                    pred_np = pred.detach().cpu().numpy().flatten()
                    targets_np = targets.detach().cpu().numpy().flatten()
                    
                    # Обратное масштабирование если нужно
                    if scaler is not None:
                        try:
                            # Восстанавливаем форму для обратного преобразования
                            pred_2d = pred_np.reshape(-1, 1)
                            targets_2d = targets_np.reshape(-1, 1)

                            pred_np = scaler.inverse_transform(pred_2d).flatten()
                            targets_np = scaler.inverse_transform(targets_2d).flatten()
                        except Exception as e:
                            print(f"Предупреждение: ошибка масштабирования: {e}")
                    
                    # Убедимся, что нет отрицательных значений
                    pred_np = np.maximum(pred_np, 0)
                    targets_np = np.maximum(targets_np, 0)

                    # УМНАЯ ПОСТОБРАБОТКА для дискретных данных
                    pred_np = postprocess_predictions(pred_np, targets_np)
                    
                    all_preds.append(pred_np)
                    all_targets.append(targets_np)

                    pbar.update(1)
                    pbar.set_postfix({"батч": batch_idx+1, "примеров": len(pred_np)})
                    
                except Exception as e:
                    print(f"Ошибка батча {batch_idx}: {e}")
                    continue
            pbar.close()
        
        if not all_preds:
            print("Нет данных для оценки!")
            return float('nan'), float('nan'), float('nan'), float('nan')
        
        # Объединяем результаты
        preds = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        print(f"Оценка завершена: {len(preds)} предсказаний")
        print(f"Диапазон предсказаний: {preds.min():.1f} - {preds.max():.1f}")
        print(f"Диапазон фактов: {y_true.min():.1f} - {y_true.max():.1f}")
        
        # ДЕТАЛЬНЫЙ АНАЛИЗ КАЧЕСТВА
        quality_metrics = analyze_predictions_quality(preds, y_true, "Evaluation")
        
        return (quality_metrics['mae'], quality_metrics['rmse'], 
                quality_metrics['r2'], quality_metrics['smape'])
        
    except Exception as e:
        print(f"Ошибка оценки: {e}")
        return float('nan'), float('nan'), float('nan'), float('nan')

def quick_test(df_train):
    """Быстрый тест на минимальных данных - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    print("   ЭКСТРЕННЫЙ ТЕСТ")
    
    # Берем данные одного магазина с достаточным количеством записей
    shop_counts = df_train.groupby('Магазин').size()
    suitable_shops = shop_counts[shop_counts >= 500].index.tolist()
    
    if not suitable_shops:
        print("Нет подходящих магазинов для теста")
        return False
        
    test_shop = suitable_shops[0]
    test_data = df_train[df_train['Магазин'] == test_shop].head(200).copy()
    
    print(f"Тестовые данные: {len(test_data)} записей из магазина {test_shop}")
    
    try:
        # Создаем ОЧЕНЬ ПРОСТОЙ датасет
        dataset = TimeSeriesDataSet(
            test_data,
            time_idx="time_idx",
            target="Количество",
            group_ids=["Магазин", "НоменклатураКод"],
            min_encoder_length=5,  # Уменьшаем для теста
            max_encoder_length=10, # Уменьшаем для теста
            min_prediction_length=3, # Уменьшаем для теста
            max_prediction_length=3,
            static_categoricals=["Магазин", "НоменклатураКод"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["Количество"],
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
        )
        
        print(f"Тестовый датасет создан: {len(dataset)} рядов")
        
        # Простой тренировочный цикл
        train_dl = make_dataloader(dataset, train=True, batch_size=4)
        
        # Проверяем один батч
        for batch_idx, batch in enumerate(train_dl):
            if batch_idx >= 1:
                break
            print(f"Тестовый батч: тип={type(batch)}")
            
            # Просто проверяем, что батч загружается
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch
                print(f"Размеры батча: x={len(x)}, y={len(y)}")
            else:
                print(f"Батч загружен успешно")
        
        print("ТЕСТ ПРОЙДЕН!")
        return True
        
    except Exception as e:
        print(f"ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_target_distribution(df_train, df_val, df_test):
    """Анализ распределения целевой переменной"""
    print("\n   АНАЛИЗ РАСПРЕДЕЛЕНИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    
    for name, df in [("Train", df_train), ("Val", df_val)]:
        if len(df) == 0:
            continue
            
        target = df['Количество'].values
        print(f"{name}:")
        print(f"  Всего записей: {len(target)}")
        print(f"  Нулевых значений: {(target == 0).sum()} ({(target == 0).sum()/len(target)*100:.1f}%)")
        print(f"  Минимум: {target.min()}")
        print(f"  Максимум: {target.max()}")
        print(f"  Среднее: {target.mean()}")
        print(f"  Медиана: {np.median(target)}")
        
        non_zero = target[target > 0]
        if len(non_zero) > 0:
            print(f"  Ненулевые значения: {len(non_zero)}")
            print(f"  Мин ненулевых: {non_zero.min()}")
            print(f"  Макс ненулевых: {non_zero.max()}")
            print(f"  Среднее ненулевых: {non_zero.mean()}")

def check_indices(df):
    """Проверка целостности индексов"""
    issues = []
    for group_name, group_data in df.groupby("Магазин"):
        group_data = group_data.sort_values("time_idx")
        time_diff = np.diff(sorted(group_data["time_idx"].unique()))
        if len(time_diff) > 0 and np.any(time_diff > 1):
            gaps = time_diff[time_diff > 1]
            issues.append(f"Пропуски в time_idx у {group_name}: {len(gaps)} пропусков")
        
        duplicate_mask = group_data.duplicated(subset=["НоменклатураКод", "time_idx"], keep=False)
        if duplicate_mask.any():
            duplicate_count = duplicate_mask.sum()
            issues.append(f"Дубликаты time_idx у {group_name}: {duplicate_count} дубликатов")
            df = df.drop_duplicates(subset=["Магазин", "НоменклатураКод", "time_idx"], keep='first')
    
    return issues, df

def get_adjusted_horizons(df):
    """Автоматическая корректировка горизонтов на основе данных"""
    # Анализ максимальной доступной длины рядов
    series_lengths = df.groupby(['Магазин', 'НоменклатураКод'])['time_idx'].agg(
        lambda x: x.max() - x.min() + 1
    )
    
    max_series_length = series_lengths.max()
    print(f"Максимальная длина ряда: {max_series_length} дней")
    
    # Рассчитываем безопасные горизонты
    safe_horizons = []
    
    # Минимальные требования: энкодер (30) + прогноз + буфер (15)
    if max_series_length >= 30 + 7 + 15:  # 52 дня
        safe_horizons.append(7)
    if max_series_length >= 30 + 14 + 15:  # 59 дней  
        safe_horizons.append(14)
    if max_series_length >= 45 + 30 + 15:  # 90 дней
        safe_horizons.append(30)
    if max_series_length >= 60 + 60 + 15:  # 135 дней
        safe_horizons.append(60)
    
    if not safe_horizons:
        # Если ничего не подходит, используем самый короткий горизонт
        min_horizon = max(7, (max_series_length - 45) // 2)  # Безопасный расчет
        safe_horizons.append(min_horizon)
    
    print(f"Безопасные горизонты прогноза: {safe_horizons}")
    return safe_horizons

def validate_data_before_training(df_train, df_val, df_test):
    """Валидация данных перед обучением"""
    print("\n   ПРОВЕРКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ")
    
    issues = []
    
    # Проверка распределения целевой переменной
    train_target = df_train['Количество'].values
    unique_values = np.unique(train_target)
    
    print(f"Уникальные значения в целевой переменной: {unique_values}")
    print(f"Распределение: {np.bincount(train_target.astype(int))}")
    
    if len(unique_values) < 2:
        issues.append("Слишком мало уникальных значений в целевой переменной")
    
    # Проверка на константные прогнозы
    if np.std(train_target) < 0.1:
        issues.append("Очень низкая дисперсия целевой переменной")
    
    # Проверка наличия всех магазинов в train
    train_shops = set(df_train['Магазин'].unique())
    if len(df_val) > 0:
        val_shops = set(df_val['Магазин'].unique())
        missing_in_train = val_shops - train_shops
        if missing_in_train:
            issues.append(f"В val есть магазины, отсутствующие в train: {missing_in_train}")
    
    if len(df_test) > 0:
        test_shops = set(df_test['Магазин'].unique())
        missing_in_train = test_shops - train_shops
        if missing_in_train:
            issues.append(f"В test есть магазины, отсутствующие в train: {missing_in_train}")
    
    if issues:
        print("ПРОБЛЕМЫ С ДАННЫМИ:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Данные прошли проверку")
    
    return len(issues) == 0

def analyze_predictions_quality(preds, y_true, horizon_name):
    """Улучшенный анализ качества прогнозов для дискретных данных"""
    print(f"\n   ДЕТАЛЬНЫЙ АНАЛИЗ КАЧЕСТВА ПРОГНОЗОВ ({horizon_name})")
    
    # Преобразуем в целые числа для дискретного анализа
    preds_int = np.round(preds).astype(int)
    y_true_int = np.round(y_true).astype(int)
    
    # Точность классификации (сколько точно угадали значение)
    exact_match = (preds_int == y_true_int).mean() * 100
    
    # Точность ±1 (сколько угадали с ошибкой ±1)
    within_one = (np.abs(preds_int - y_true_int) <= 1).mean() * 100
    
    print(f"Точное совпадение: {exact_match:.1f}%")
    print(f"Совпадение ±1: {within_one:.1f}%")
    
    # Матрица ошибок для дискретных значений
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_int, preds_int, labels=[1, 2, 3, 4])
    
    print("\nМатрица ошибок:")
    print("     Прогноз→")
    print("Факт↓  1  2  3  4")
    for i, true_val in enumerate([1, 2, 3, 4]):
        row = f"  {true_val}   "
        for j, pred_val in enumerate([1, 2, 3, 4]):
            row += f"{cm[i,j]:2d} "
        print(row)
    
    # Анализ смещения
    bias = preds_int - y_true_int
    under_pred = (bias < 0).sum()
    over_pred = (bias > 0).sum()
    correct = (bias == 0).sum()
    
    print(f"\nСмещение прогнозов:")
    print(f"  Занижено: {under_pred} ({under_pred/len(preds)*100:.1f}%)")
    print(f"  Завышено: {over_pred} ({over_pred/len(preds)*100:.1f}%)") 
    print(f"  Точно: {correct} ({correct/len(preds)*100:.1f}%)")
    
    # Метрики регрессии
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    
    if len(np.unique(y_true)) > 1 and np.var(y_true) > 1e-10:
        r2 = r2_score(y_true, preds)
    else:
        r2 = float('nan')
        
    smape_val = 100 * np.mean(2 * np.abs(preds - y_true) / (np.abs(y_true) + np.abs(preds) + 1e-8))
    
    print(f"\nМетрики регрессии:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  SMAPE: {smape_val:.1f}%")
    
    return {
        'exact_match': exact_match,
        'within_one': within_one,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'smape': smape_val
    }


def run_improvement_pipeline(df_train, df_val, df_test):
    """Запуск всего пайплайна улучшений"""
    print("\n" + "="*80)
    print("ЗАПУСК ПАЙПЛАЙНА УЛУЧШЕНИЯ МЕТРИК")
    print("="*80)
    
    # 1. Ансамблирование
    try:
        from ensemble import run_ensemble_experiment
        ensemble_r2, ensemble_mae = run_ensemble_experiment(df_train, df_val, df_test)
        
        print(f"\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
        print(f"Ensemble:     R² = {ensemble_r2:.4f} (val) {'ОК' if ensemble_r2 > 0.1 else 'ERR'}")
        
    except Exception as e:
        print(f"Ансамблирование не удалось: {e}")
    
    # 2. Добавление супер-фич к TFT
    try:
        print("\nДОБАВЛЕНИЕ СУПЕР-ФИЧ К TFT...")
        df_train_super = create_super_features(df_train.copy())
        df_val_super = create_super_features(df_val.copy())
        
        # Здесь можно переобучить TFT с новыми фичами
        print("Супер-фичи добавлены! Можно переобучить TFT")
        
    except Exception as e:
        print(f"Добавление супер-фич не удалось: {e}")
    

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # import glob
    # old_models = glob.glob(os.path.join(MODEL_DIR, "*.ckpt")) + glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    # for model_file in old_models:
    #     try:
    #         os.remove(model_file)
    #         print(f"Удалена старая модель: {model_file}")
    #     except:
    #         pass
    torch.cuda.empty_cache()
    optimize_memory()

    # Настройка перехвата ошибок
    sys.excepthook = lambda exctype, value, traceback: error_logger.error(
        f"Необработанное исключение: {exctype.__name__}: {value}", exc_info=True
    )
    
    logging.info("Загрузка данных...")
    
    # Получаем доступные магазины и категории
    available_shops, available_gaps = get_available_shops_and_gaps(DB_PATH)

    # Тестовый режим
    TEST_MODE = False

    if TEST_MODE:
        selected_shops = available_shops[:5]
        selected_gaps = available_gaps[:5]
        print(f"ТЕСТОВЫЙ РЕЖИМ: {len(selected_shops)} магазинов, {len(selected_gaps)} категорий")
    else:
        selected_shops = available_shops
        selected_gaps = available_gaps

    # Загрузка и подготовка данных
    df = prepare_data_from_db(DB_PATH, selected_shops, selected_gaps)
    aggressive_memory_cleanup()

    print("   ФИНАЛЬНАЯ ПРОВЕРКА ТИПОВ ДАННЫХ")
    categorical_check = ['weekday', 'day', 'promoFlag', 'is_holiday_day', 'Месяц', 'ГОД']
    for col in categorical_check:
        if col in df.columns:
            dtype = df[col].dtype
            sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
            print(f"{col}: тип={dtype}, пример={sample}, тип_значения={type(sample)}")
    
    shop_stats = check_shop_data_distribution(df)
    
    # Агрегация данных
    # df = aggregate_daily_data(df)
    # aggressive_memory_cleanup()
    # Разделение данных
    df_train, df_val, df_test = split_by_time(df)

    print("\n" + "="*50)
    print("ОБРАБОТКА СПАРС ДАННЫХ")
    print("="*50)

    # Обрабатываем спарс данные
    # df_train, df_val, df_test = handle_sparse_data_simple(df_train, df_val, df_test) 

    if len(df_train) < 1000:
        print("После фильтрации осталось мало данных, используем исходные...")
        df_train, df_val, df_test = split_by_time(df)

    print("   ПРОВЕРКА РАЗДЕЛЕНИЯ ДАННЫХ")
    print(f"Train размер: {len(df_train)}")
    print(f"Val размер: {len(df_val)}") 
    print(f"Test размер: {len(df_test)}")

    analyze_target_distribution(df_train, df_val, df_test)

    if len(df_train) == 0:
        raise ValueError("Нет данных для обучения!")

    print("Группы в train:", df_train["Магазин"].unique())
    print("Группы в val:", df_val["Магазин"].unique() if len(df_val) > 0 else "Нет данных")
    print("Группы в test:", df_test["Магазин"].unique() if len(df_test) > 0 else "Нет данных")

    # Проверяем, что все группы из val и test есть в train
    train_groups = set(df_train["Магазин"].unique())
    if len(df_val) > 0:
        val_groups = set(df_val["Магазин"].unique())
        missing_in_train = val_groups - train_groups
        if missing_in_train:
            print(f"Предупреждение: в val есть группы, отсутствующие в train: {missing_in_train}")
            df_val = df_val[df_val["Магазин"].isin(train_groups)]

    if len(df_test) > 0:
        test_groups = set(df_test["Магазин"].unique())
        missing_in_train = test_groups - train_groups
        if missing_in_train:
            print(f"Предупреждение: в test есть группы, отсутствующие в train: {missing_in_train}")
            df_test = df_test[df_test["Магазин"].isin(train_groups)]

    print("После фильтрации:")
    print(f"Train: {len(df_train)} записей, группы: {df_train['Магазин'].unique()}")
    print(f"Val: {len(df_val)} записей, группы: {df_val['Магазин'].unique() if len(df_val) > 0 else 'Нет'}")
    print(f"Test: {len(df_test)} записей, группы: {df_test['Магазин'].unique() if len(df_test) > 0 else 'Нет'}")

    # Масштабирование
    # df_train, df_val, df_test, scaler = scale_numeric(df_train, df_val, df_test)

    print("Проверка целостности данных...")
    print(f"Всего записей: {len(df)}")
    print(f"Уникальных магазинов: {df['Магазин'].nunique()}")
    print(f"Диапазон дат: {df['Дата'].min()} - {df['Дата'].max()}")
    print(f"Диапазон time_idx: {df['time_idx'].min()} - {df['time_idx'].max()}")

    # Проверка индексов
    index_issues, df_train = check_indices(df_train)
    if index_issues:
        print("Проблемы с индексами:")
        for issue in index_issues:
            print(f"  - {issue}")

    # Быстрый тест
    print("   ЗАПУСК ЭКСТРЕННОГО ТЕСТА")
    if not quick_test(df_train):
        print("ОСНОВНОЕ ОБУЧЕНИЕ ПРЕРВАНО - ТЕСТ НЕ ПРОЙДЕН")
        sys.exit(1)

    results = []
    # Проверка доступных горизонтов:
    # HORIZONS = get_adjusted_horizons(df)
    
    # if not HORIZONS:
    #     print("!!! Невозможно определить безопасные горизонты прогноза!")
    #     print("Рекомендуется собрать больше данных или уменьшить требования к горизонту")
    #     sys.exit(1)
    
    # print(f"Используемые горизонты: {HORIZONS}")

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"=======ОБУЧЕНИЕ ДЛЯ ГОРИЗОНТА {horizon} ДНЕЙ")
        print(f"{'='*60}")
        
        if horizon == 60:
            if not check_60_days_viability(df):
                series_lengths = analyze_series_lengths_detailed(df)
                print("Пропускаем горизонт 60 дней - недостаточно данных")
                continue

        if not analyze_data_quality(df, horizon):
            print(f"ВНИМАНИЕ: Данные могут быть недостаточны для горизонта {horizon} дней")
            print("Рекомендуется уменьшить горизонт прогноза или собрать больше данных")

        clear_memory()
        
        horizon_params = get_adaptive_hyperparameters(df_train, horizon)
        logging.info(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        print(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        
        # Анализ данных
        print("   АНАЛИЗ ДОСТУПНЫХ ДАННЫХ")
        total_groups = df['Магазин'].nunique()
        total_items = df['НоменклатураКод'].nunique()
        
        group_stats = df.groupby('Магазин')['time_idx'].agg(['min', 'max', 'count']).reset_index()
        group_stats['length'] = group_stats['max'] - group_stats['min'] + 1
        
        print(f"Всего групп: {total_groups}")
        print(f"Всего товаров: {total_items}")
        print(f"Средняя длина ряда: {group_stats['length'].mean():.1f} дней")
        print(f"Минимальная длина: {group_stats['length'].min()} дней")
        print(f"Максимальная длина: {group_stats['length'].max()} дней")

        # Разделяем данные
        df_train, df_val, df_test = split_by_time(df)
        
        logging.info(f"Данные для горизонта {horizon}: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        print(f"Данные: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # Масштабирование
        df_train, df_val, df_test, scaler = scale_numeric(df_train, df_val, df_test)

        # Создаем datasets
        try:
            print("   СОЗДАНИЕ ДАТАСЕТА")
            if horizon == 30:
                train_ds = create_simple_dataset(df_train, prediction_length=horizon)
                actual_horizon = horizon
            if horizon == 60:
                train_ds, val_ds, test_ds, actual_horizon = create_smart_60d_dataset_all_data(df_train, df_val, df_test, horizon)
        
                if train_ds is None:
                    print("Не удалось создать датасет для 60 дней")
                    continue
            
            print(f"Фактический горизонт: {actual_horizon} дней")
            import pickle
            import os

            SAVE_DIR = "saved_data"
            os.makedirs(SAVE_DIR, exist_ok=True)

            # Сохраняем датасет
            train_ds_path = os.path.join(SAVE_DIR, f"train_ds_{horizon}d.pkl")
            with open(train_ds_path, "wb") as f:
                pickle.dump(train_ds, f)
            print(f"Train dataset сохранен: {train_ds_path}")

            # Сохраняем scaler
            scaler_path = os.path.join(SAVE_DIR, f"scaler_{horizon}d.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Scaler сохранен: {scaler_path}")

            # Создаем val_ds и test_ds
            val_ds = None
            if len(df_val) > 0 and train_ds is not None:
                val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True)
            
            test_ds = None  
            # if len(df_test) > 0 and train_ds is not None:
            #     test_ds = TimeSeriesDataSet.from_dataset(train_ds, df_test, predict=True)
                
            print("Все датасеты созданы успешно!")
                
        except Exception as e:
            print(f"Финальная ошибка: {e}")
            print("Пропускаем этот горизонт...")
            continue

#         if not validate_data_before_training(df_train, df_val, df_test):
#             print("Обнаружены проблемы с данными")

#         if horizon == 60:
#             # Пробуем файн-тюнинг сначала
#             model = fine_tune_60d_model(df_train, df_val, df_test)
#             if model is not None:
#                 print("Успешно использован файн-тюнинг для 60 дней!")
#                 # Пропускаем обычное обучение
#                 continue
#             else:
#                 print("Файн-тюнинг не удался, обучаем с нуля...")

#         # Обучение
#         model, train_dl, val_dl = train_one_horizon(train_ds, val_ds, None, prediction_length=horizon, horizon_params=horizon_params)

#         if model is None or train_dl is None:
#             print(f"Обучение не удалось для горизонта {horizon}, пропускаем...")
#             continue

#         if model is not None:
#             print("---   БЫСТРАЯ ОЦЕНКА МОДЕЛИ")

#             # УБЕДИТЕСЬ, что модель на правильном устройстве
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model.to(device)
#             print(f"Модель подготовлена на устройстве: {device}")
            
#             mae, rmse, r2, smape = float('nan'), float('nan'), float('nan'), float('nan')
#             mae_val, rmse_val, r2_val, smape_val = float('nan'), float('nan'), float('nan'), float('nan')
#             mae_test, rmse_test, r2_test, smape_test = float('nan'), float('nan'), float('nan'), float('nan')
            
#             # БЫСТРАЯ оценка на тренировочных данных
#             if train_dl is not None:
#                 print("Быстрая оценка на тренировочных данных...")
#                 try:
#                     mae, rmse, r2, smape = evaluate_metrics(
#                         model, train_dl, scaler, max_batches=5
#                     )
#                     print(f"TRAIN (быстрая): MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} | SMAPE={smape:.2f}%")
                    
#                     # ВИЗУАЛИЗАЦИЯ ПРОГНОЗА
#                     print("\n--- Генерация графиков прогнозов ---")
#                     try:
#                         print("\n--- Отладка структуры данных ---")
#                         vz.debug_batch_structure(train_dl)
                        
#                         print("\n--- Быстрая визуализация ---")
#                         vz.quick_visualization_fixed(model, train_dl, scaler, horizon=horizon)
                        
#                         print("\n--- Полная визуализация ---")
#                         predictions, actuals = vz.plot_predictions_vs_actual_tft_fixed(
#                             model, train_dl, scaler, horizon=horizon, n_examples=3
#                         )
                        
#                     except Exception as e:
#                         print(f"Ошибка визуализации: {e}")
#                         import traceback
#                         traceback.print_exc()
                        
#                 except Exception as e:
#                     print(f"Ошибка оценки на train: {e}")

#             # БЫСТРАЯ оценка на валидации
#             if val_dl is not None:
#                 print("Быстрая оценка на валидации...")
#                 try:
#                     mae_val, rmse_val, r2_val, smape_val = evaluate_metrics(
#                         model, val_dl, scaler, max_batches=3
#                     )
#                     print(f"VAL (быстрая): MAE={mae_val:.4f} | RMSE={rmse_val:.4f} | R2={r2_val:.4f} | SMAPE={smape_val:.2f}%")
#                 except Exception as e:
#                     print(f"Ошибка оценки на val: {e}")
#                     mae_val, rmse_val, r2_val, smape_val = float('nan'), float('nan'), float('nan'), float('nan')
#             else:
#                 mae_val, rmse_val, r2_val, smape_val = float('nan'), float('nan'), float('nan'), float('nan')

#             # БЫСТРАЯ оценка на тесте
#             # if test_dl is not None:
#             #     print("Быстрая оценка на тесте...")
#             #     try:
#             #         mae_test, rmse_test, r2_test, smape_test = evaluate_metrics(
#             #             model, test_dl, scaler, max_batches=3
#             #         )
#             #         print(f"TEST (быстрая): MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R2={r2_test:.4f} | SMAPE={smape_test:.2f}%")
#             #     except Exception as e:
#             #         print(f"Ошибка оценки на test: {e}")
#             #         mae_test, rmse_test, r2_test, smape_test = float('nan'), float('nan'), float('nan'), float('nan')
#             # else:
#             #     mae_test, rmse_test, r2_test, smape_test= float('nan'), float('nan'), float('nan'), float('nan')

#             # Сохраняем результаты
#             results.append({
#                 "horizon_days": horizon,
#                 "MAE_TRAIN": mae, "RMSE_train": rmse, "R2_train": r2, "SMAPE_train": smape,
#                 "MAE_VAL": mae_val, "RMSE_val": rmse_val, "R2_val": r2_val, "SMAPE_val": smape_val,
#                 "MAE_TEST": mae_test, "RMSE_test": rmse_test, "R2_test": r2_test, "SMAPE_test": smape_test,
#                 "batch_size": horizon_params["batch_size"],
#                 "accumulate_grad_batches": horizon_params.get("accumulate_grad_batches", 1),
#                 "effective_batch_size": horizon_params["batch_size"] * horizon_params.get("accumulate_grad_batches", 1),
#                 "groups_used": total_groups,
#                 "items_used": total_items
#             })

#             # Сохранение модели
#             model_path = os.path.join(MODEL_DIR, f"tft_{horizon}d.pth")
#             torch.save(model.state_dict(), model_path)
#             logging.info(f"Модель сохранена: {model_path}")
            
#             # Очистка памяти
#             del model, train_ds, val_ds, train_dl, val_dl
#             clear_memory()
#         else:
#             logging.error(f"Обучение для горизонта {horizon} не удалось")

#     run_improvement_pipeline(df_train, df_val, df_test)

#     try:
#         from experiments import run_experiments_from_main
        
#         print("\n" + "="*80)
#         print("ЗАПУСК ЭКСПЕРИМЕНТАЛЬНЫХ МЕТОДОВ")
#         print("="*80)
        
#         # Запускаем эксперименты ПОСЛЕ основного обучения
#         experimental_models = run_experiments_from_main(df_train, df_val, df_test, scaler)
        
#         if experimental_models:
#             print("Эксперименты завершены успешно!")
#         else:
#             print("Эксперименты не дали результатов")
            
#     except ImportError as e:
#         print(f"Не удалось импортировать experiments: {e}")
#     except Exception as e:
#         print(f"Ошибка в экспериментах: {e}")



# # сброс кеша: python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect(); print('Память очищена')"
# # Starts: Z:\TF_GPU\venv\Scripts\activate --> python forecasting_checks_copy.py > full_output.log 2>&1


