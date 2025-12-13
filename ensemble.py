import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RetailEnsemble:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def create_advanced_features(self, df):
        """Создание продвинутых фич для ансамбля"""
        print("Создание продвинутых фич для ансамбля...")
        
        # Базовые фичи
        df['year'] = df['Дата'].dt.year
        df['month'] = df['Дата'].dt.month
        df['day_of_year'] = df['Дата'].dt.dayofyear
        df['week_of_year'] = df['Дата'].dt.isocalendar().week
        df['quarter'] = df['Дата'].dt.quarter
        
        # Сезонные фичи высокого порядка
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        # Фичи пиковой активности
        df['is_weekend'] = df['Дата'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = (df['Дата'].dt.day == 1).astype(int)
        df['is_month_end'] = (df['Дата'].dt.day == df['Дата'].dt.days_in_month).astype(int)
        
        # Взаимодействия
        df['weekend_holiday'] = df['is_weekend'] * (df['is_holiday_day'] == '1').astype(int)
        df['promo_weekend'] = (df['promoFlag'] == '1').astype(int) * df['is_weekend']
        
        # Скользящие статистики
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            df[f'rolling_max_{window}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        
        # Лаги разных порядков
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f'lag_{lag}'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].shift(lag)
        
        # Разности
        df['diff_1'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].diff(1)
        df['diff_7'] = df.groupby(['Магазин', 'НоменклатураКод'])['Количество'].diff(7)
        
        # Статистики по группам
        df['item_avg_sales'] = df.groupby('НоменклатураКод')['Количество'].transform('mean')
        df['item_median_sales'] = df.groupby('НоменклатураКод')['Количество'].transform('median')
        df['store_avg_sales'] = df.groupby('Магазин')['Количество'].transform('mean')
        
        print(f"Создано {len([col for col in df.columns if col not in ['Дата', 'Количество']])} фич")
        return df
    
    def prepare_ensemble_features(self, df):
        """Подготовка фич для ансамбля"""
        # Выбираем числовые и категориальные фичи
        numeric_features = [
            'time_idx', 'year', 'month', 'day_of_year', 'week_of_year', 'quarter',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_weekend',
            'is_month_start', 'is_month_end', 'weekend_holiday', 'promo_weekend'
        ]
        
        # Добавляем скользящие статистики
        for window in [7, 14, 30]:
            numeric_features.extend([
                f'rolling_mean_{window}', f'rolling_std_{window}', f'rolling_max_{window}'
            ])
        
        # Добавляем лаги
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            numeric_features.append(f'lag_{lag}')
        
        numeric_features.extend(['diff_1', 'diff_7', 'item_avg_sales', 'item_median_sales', 'store_avg_sales'])
        
        # Отбираем только существующие фичи
        available_features = [f for f in numeric_features if f in df.columns]
        
        # Заполняем пропуски
        for feature in available_features:
            if df[feature].isnull().any():
                if 'lag' in feature or 'diff' in feature or 'rolling' in feature:
                    df[feature] = df[feature].fillna(0)
                else:
                    df[feature] = df[feature].fillna(df[feature].median())
        
        return df[available_features], available_features
    
    def train_ensemble(self, df_train, df_val, target_col='Количество'):
        """Обучение ансамбля моделей"""
        print("=== ОБУЧЕНИЕ АНСАМБЛЯ МОДЕЛЕЙ ===")
        
        # Создаем фичи
        df_train = self.create_advanced_features(df_train)
        df_val = self.create_advanced_features(df_val)
        
        # Подготавливаем фичи
        X_train, features = self.prepare_ensemble_features(df_train)
        X_val, _ = self.prepare_ensemble_features(df_val)
        
        y_train = df_train[target_col].values
        y_val = df_val[target_col].values
        
        print(f"Формы данных: X_train {X_train.shape}, y_train {y_train.shape}")
        
        # Ансамбль моделей
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        # Обучение и оценка
        ensemble_predictions = {}
        feature_importance = {}
        
        for name, model in self.models.items():
            print(f"Обучение {name}...")
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Метрики
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            print(f"{name:15} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
            
            ensemble_predictions[name] = y_pred_val
            
            # Важность фич (если доступно)
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = dict(zip(features, model.feature_importances_))
        
        # Взвешенный ансамбль
        print("\n=== ВЗВЕШЕННЫЙ АНСАМБЛЬ ===")
        weights = {
            'random_forest': 0.5,
            'gradient_boosting': 0.3, 
            'linear': 0.2
        }
        
        y_ensemble = sum(weights[name] * ensemble_predictions[name] for name in weights.keys())
        
        ensemble_r2 = r2_score(y_val, y_ensemble)
        ensemble_mae = mean_absolute_error(y_val, y_ensemble)
        
        print(f"АНСАМБЛЬ | Val R²: {ensemble_r2:.4f} | Val MAE: {ensemble_mae:.4f}")
        
        self.feature_importance = feature_importance
        return ensemble_r2, ensemble_mae
    
    def get_feature_importance(self, top_n=15):
        """Анализ важности фич"""
        if not self.feature_importance:
            print("Нет данных о важности фич")
            return
        
        print(f"\n=== ТОП-{top_n} ВАЖНЫХ ФИЧ ===")
        
        # Объединяем важности из всех моделей
        combined_importance = {}
        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                if feature not in combined_importance:
                    combined_importance[feature] = []
                combined_importance[feature].append(importance)
        
        # Усредняем
        avg_importance = {feature: np.mean(importances) for feature, importances in combined_importance.items()}
        
        # Сортируем по важности
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        for feature, importance in sorted_features:
            print(f"{feature:30}: {importance:.4f}")

# Интеграция с основной моделью
def run_ensemble_experiment(df_train, df_val, df_test):
    """Запуск ансамбля для сравнения с TFT"""
    ensemble = RetailEnsemble()
    
    print("ЗАПУСК АНСАМБЛЯ ДЛЯ СРАВНЕНИЯ С TFT")
    
    # Обучаем на train/val
    ensemble_r2, ensemble_mae = ensemble.train_ensemble(df_train, df_val)
    
    # # Тестируем на test
    # df_test_enhanced = ensemble.create_advanced_features(df_test.copy())
    # X_test, _ = ensemble.prepare_ensemble_features(df_test_enhanced)
    # y_test = df_test_enhanced['Количество'].values
    
    # test_predictions = {}
    # for name, model in ensemble.models.items():
    #     y_pred_test = model.predict(X_test)
    #     test_r2 = r2_score(y_test, y_pred_test)
    #     test_mae = mean_absolute_error(y_test, y_pred_test)
    #     test_predictions[name] = y_pred_test
    #     print(f"{name:15} | TEST R²: {test_r2:.4f} | TEST MAE: {test_mae:.4f}")
    
    print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ (только val) ===")
    
    # Создаем фичи для val
    df_val_enhanced = ensemble.create_advanced_features(df_val.copy())
    X_val, _ = ensemble.prepare_ensemble_features(df_val_enhanced)
    y_val = df_val_enhanced['Количество'].values
    
    val_predictions = {}
    for name, model in ensemble.models.items():
        y_pred_val = model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_predictions[name] = y_pred_val
        print(f"{name:15} | VAL R²: {val_r2:.4f} | VAL MAE: {val_mae:.4f}")
    
    # Ансамбль на val
    weights = {'random_forest': 0.5, 'gradient_boosting': 0.3, 'linear': 0.2}
    y_ensemble_val = sum(weights[name] * val_predictions[name] for name in weights.keys())
    
    ensemble_val_r2 = r2_score(y_val, y_ensemble_val)
    ensemble_val_mae = mean_absolute_error(y_val, y_ensemble_val)
    
    print(f"\nАНСАМБЛЬ НА VAL | R²: {ensemble_val_r2:.4f} | MAE: {ensemble_val_mae:.4f}")
    
    # СОХРАНЯЕМ МОДЕЛИ АНСАМБЛЯ
    print("\n=== СОХРАНЕНИЕ МОДЕЛЕЙ АНСАМБЛЯ ===")
    import joblib
    import os
    
    ensemble_dir = "ensemble_models"
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for name, model in ensemble.models.items():
        model_path = os.path.join(ensemble_dir, f"ForEnsemble_{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Сохранена модель: {model_path}")
    
    # Сохраняем сам ансамбль с весами
    ensemble_data = {
        'models': {name: type(model).__name__ for name, model in ensemble.models.items()},
        'weights': weights,
        'feature_importance': ensemble.feature_importance
    }
    ensemble_path = os.path.join(ensemble_dir, "ensemble_config.pkl")
    joblib.dump(ensemble_data, ensemble_path)
    print(f"Сохранена конфигурация ансамбля: {ensemble_path}")
    
    # Анализ важности фич
    ensemble.get_feature_importance(top_n=15)
    
    return ensemble_val_r2, ensemble_val_mae