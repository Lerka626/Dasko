import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import os
import argparse
import pickle
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import sqlite3
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import holidays
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAVE_DIR = os.path.join(BASE_DIR, "saved_data")
DB_PATH = os.path.join(BASE_DIR, "only_aclo_final.db")
OUT_DIR = MODEL_DIR


def load_data_from_db():
    """Загружает данные из базы данных"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    
    # ПРОСТОЙ ЗАПРОС - все колонки уже в sales_data!
    df = pd.read_sql_query(
        """
        SELECT 
            Дата, 
            Магазин, 
            НоменклатураКод, 
            Количество, 
            promoFlag, 
            is_holiday_day,
            Категория,
            GAP
        FROM sales_data
        ORDER BY Магазин, НоменклатураКод, Дата
        """,
        conn,
    )
    conn.close()
    
    print(f"Загружено {len(df)} строк из sales_data")
    print(f"Колонки: {df.columns.tolist()}")
    
    return df

def preprocess_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных"""
    df = df.copy()
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    df["Количество"] = pd.to_numeric(df["Количество"], errors="coerce").fillna(0.0)

    df = (
        df.groupby(["Магазин", "НоменклатураКод", "Дата", "Категория", "GAP"], as_index=False)
        .agg({"Количество": "sum", "promoFlag": "first", "is_holiday_day": "first"})
        .sort_values(["Магазин", "НоменклатураКод", "Дата"])
        .reset_index(drop=True)
    )

    df["weekday"] = df["Дата"].dt.weekday.astype(str)
    df["promoFlag"] = df["promoFlag"].fillna(0).astype(str)

    ru_holidays = holidays.RU()
    df["is_holiday_day"] = df["Дата"].apply(
        lambda d: "1"
        if (d in ru_holidays or (d.month == 12 and d.day >= 18))
        else "0"
    )

    return df


def compute_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет time_idx для временных рядов"""
    df = df.copy()
    df["time_idx"] = df.groupby(["Магазин", "НоменклатураКод"]).cumcount()
    return df


def filter_data_by_selection(
    df: pd.DataFrame,
    shops: Optional[List[str]] = None,
    items: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Фильтрует данные по выбранным магазинам и товарам
    
    Args:
        df: Исходный DataFrame
        shops: Список магазинов для фильтрации
        items: Список кодов товаров для фильтрации
        
    Returns:
        Отфильтрованный DataFrame
    """
    filtered_df = df.copy()
    
    # Фильтрация по магазинам
    if shops:
        filtered_df = filtered_df[filtered_df["Магазин"].isin(shops)]
        print(f"  Отфильтровано по магазинам: {len(shops)} магазинов")
    
    # Фильтрация по товарам
    if items:
        filtered_df = filtered_df[filtered_df["НоменклатураКод"].isin(items)]
        print(f"  Отфильтровано по товарам: {len(items)} товаров")
    
    return filtered_df

def build_future_df(df, horizon, max_encoder_length):
    """Строит DataFrame для прогнозирования"""
    ru_holidays = holidays.RU()
    frames = []

    for (shop, item), g in df.groupby(["Магазин", "НоменклатураКод"]):
        g = g.sort_values("Дата")
        enc = g.tail(max_encoder_length)

        last_date = g["Дата"].iloc[-1]
        last_idx = g["time_idx"].iloc[-1]

        future_dates = pd.date_range(
            last_date + timedelta(days=1), periods=horizon, freq="D"
        )

        future = pd.DataFrame(
            {
                "Дата": future_dates,
                "Магазин": shop,
                "НоменклатураКод": item,
                "Количество": 0.0,
                "time_idx": np.arange(last_idx + 1, last_idx + 1 + horizon),
            }
        )

        future["weekday"] = future["Дата"].dt.weekday.astype(str)
        future["promoFlag"] = "0"
        future["is_holiday_day"] = future["Дата"].apply(
            lambda d: "1"
            if (d in ru_holidays or (d.month == 12 and d.day >= 18))
            else "0"
        )

        frames.append(pd.concat([enc, future], ignore_index=True))

    return pd.concat(frames, ignore_index=True)

def robust_load_state_dict(model, path):
    """Загружает веса модели с обработкой несовпадений"""
    state = torch.load(path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    model_sd = model.state_dict()
    new_sd = {}

    for k, v in model_sd.items():
        if k in state and state[k].shape == v.shape:
            new_sd[k] = state[k]
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd)
    return model


def predict_cpu(model, dataloader):
    """Выполняет прогнозирование на CPU"""
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = model(x)
            preds.append(out.prediction.cpu())

    return torch.cat(preds, dim=0).numpy().reshape(-1)


def run(
    horizon=30,
    shops: Optional[List[str]] = None,
    items: Optional[List[str]] = None,
    batch_size=256
) -> pd.DataFrame:
    """
    Основная функция запуска прогнозирования
    
    Args:
        horizon: Горизонт прогнозирования (30 или 60 дней)
        shops: Список магазинов для фильтрации
        items: Список кодов товаров для фильтрации (получены по категориям)
        batch_size: Размер батча для прогнозирования
        
    Returns:
        DataFrame с результатами прогноза
    """
    assert horizon in (30, 60), "Горизонт должен быть 30 или 60 дней"
    
    print(f"Запуск прогноза с параметрами:")
    print(f"  Горизонт: {horizon} дней")
    print(f"  Магазины: {len(shops) if shops else 'все'}")
    print(f"  Товары: {len(items) if items else 'все'}")
    
    # ---- Загружаем тренировочный dataset ----
    train_ds_path = os.path.join(SAVE_DIR, f"train_ds_{horizon}d.pkl")
    if not os.path.exists(train_ds_path):
        raise FileNotFoundError(f"Не найден тренировочный dataset: {train_ds_path}")
    
    with open(train_ds_path, "rb") as f:
        train_ds: TimeSeriesDataSet = pickle.load(f)
    
    print(f"Загружен тренировочный dataset, max_encoder_length: {train_ds.max_encoder_length}")
    
    # ---- Загружаем данные ----
    print("Загрузка данных из БД...")
    df = load_data_from_db()
    df = preprocess_sales(df)
    df = compute_time_idx(df)
    
    # ---- Применяем фильтры ----
    print("Применение фильтров...")
    df = filter_data_by_selection(df, shops, items)
    
    if df.empty:
        print("Нет данных после применения фильтров")
        return pd.DataFrame()
    
    print(f"Осталось данных: {len(df)} строк, {df['НоменклатураКод'].nunique()} уникальных товаров")
    
    # ---- Строим future DataFrame ----
    print("Построение future DataFrame...")
    future_df = build_future_df(
        df, horizon=horizon, max_encoder_length=train_ds.max_encoder_length
    )
    
    # Приведение категориальных переменных
    categorical_cols = ["Магазин", "НоменклатураКод", "weekday", "promoFlag", "is_holiday_day"]
    for c in categorical_cols:
        if c in future_df.columns:
            future_df[c] = future_df[c].astype(str).astype("category")
    
    # ---- Создаем dataset для прогнозирования ----
    print("Создание predict dataset...")
    predict_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        future_df,
        predict=True,
        stop_randomization=True,
    )
    
    predict_dl = predict_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    
    # ---- Загружаем модель ----
    print("Загрузка модели...")
    model = TemporalFusionTransformer.from_dataset(train_ds)
    weight_path = os.path.join(MODEL_DIR, f"tft_{horizon}d.pth")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Не найден файл весов модели: {weight_path}")
    
    model = robust_load_state_dict(model, weight_path)
    
    # ---- Выполняем прогнозирование ----
    print("Прогнозирование...")
    raw_preds = predict_cpu(model, predict_dl)
    
    # Обработка результатов
    h = horizon
    raw_preds = raw_preds.reshape(-1, h)
    
    # Группируем по магазинам и товарам
    group_cols = ["Магазин", "НоменклатураКод"]
    groups = (
        future_df[group_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    last_preds = raw_preds[-len(groups):]
    
    result_frames = []
    
    for i, row in groups.iterrows():
        shop = row["Магазин"]
        item = row["НоменклатураКод"]
        
        g = future_df[
            (future_df["Магазин"] == shop)
            & (future_df["НоменклатураКод"] == item)
        ].sort_values("time_idx")
        
        g_future = g.tail(h).copy()
        g_future["pred_Количество"] = np.maximum(last_preds[i], 0)
        
        result_frames.append(g_future)
    
    out = pd.concat(result_frames, ignore_index=True)
    
    # ---- Сохраняем результаты ----
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"predictions_{horizon}d_{timestamp}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    
    print(f"Прогноз сохранен: {path}")
    print(f"   Всего записей: {len(out)}")
    print(f"   Уникальных товаров: {out['НоменклатураКод'].nunique()}")
    print(f"   Уникальных магазинов: {out['Магазин'].nunique()}")
    
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогнозирование продаж")
    parser.add_argument("--horizon", type=int, default=30, choices=[30, 60], help="Горизонт прогноза")
    parser.add_argument("--shops", type=str, help="Список магазинов через запятую")
    parser.add_argument("--items", type=str, help="Список кодов товаров через запятую")
    parser.add_argument("--batch", type=int, default=256, help="Размер батча")
    
    args = parser.parse_args()
    
    # Парсинг списков
    shops_list = args.shops.split(",") if args.shops else None
    items_list = args.items.split(",") if args.items else None
    
    df = run(
        horizon=args.horizon,
        shops=shops_list,
        items=items_list,
        batch_size=args.batch,
    )
    
    if not df.empty:
        print("\nПервые 10 строк результата:")
        print(df[["Магазин", "НоменклатураКод", "Дата", "pred_Количество"]].head(10))
    else:
        print("Результат пуст")
