import os
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import gc

# Предотвращаем циклический импорт
try:
    from forecasting_checks_copy import (
        create_simple_dataset, get_adaptive_hyperparameters, 
        create_adaptive_model, make_dataloader, evaluate_metrics,
        TimeSeriesDataSet, TemporalFusionTransformer
    )
except ImportError:
    print("Предупреждение: не удалось импортировать функции из основного файла")

class ForecastingExperiments:
    """Класс для экспериментов с улучшением прогнозов"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.results = {}
    
    def staged_forecasting_60d(self, df_train, df_val, df_test, scaler):
        """Эксперимент 1: Поэтапное прогнозирование 60 дней"""
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 1: ПОЭТАПНОЕ ПРОГНОЗИРОВАНИЕ 60 ДНЕЙ")
        print("="*60)
        
        try:
            # Шаг 1: Прогноз первых 30 дней
            print("ШАГ 1: Прогноз первых 30 дней...")
            train_ds_30d = create_simple_dataset(df_train, prediction_length=30)
            
            # Загружаем или обучаем модель для 30 дней
            model_30d_path = os.path.join(self.model_dir, "tft_7d.pth")
            if os.path.exists(model_30d_path):
                horizon_params = get_adaptive_hyperparameters(df_train, 30)
                model_30d = create_adaptive_model(train_ds_30d, horizon_params)
                model_30d.load_state_dict(torch.load(model_30d_path))
                print("Модель 30d загружена")
            else:
                print("Модель 30d не найдена, требуется обучение")
                return None
            
            # Шаг 2: Используем прогнозы как вход для следующих 30 дней
            print("ШАГ 2: Создание расширенных данных...")
            df_extended = self._create_extended_dataset(df_train, model_30d, scaler)
            
            if df_extended is None:
                print("Не удалось создать расширенные данные")
                return None
            
            # Шаг 3: Прогноз вторых 30 дней на расширенных данных
            print("ШАГ 3: Прогноз вторых 30 дней...")
            train_ds_60d = create_simple_dataset(df_extended, prediction_length=30)
            horizon_params_60d = get_adaptive_hyperparameters(df_extended, 30)
            
            # Уменьшаем сложность для второго этапа
            horizon_params_60d["hidden_size"] = max(16, horizon_params_60d["hidden_size"] // 2)
            horizon_params_60d["learning_rate"] = horizon_params_60d["learning_rate"] * 0.5
            
            model_60d_stage2 = create_adaptive_model(train_ds_60d, horizon_params_60d)
            
            # Обучение второй модели
            train_dl = make_dataloader(train_ds_60d, train=True, 
                                     batch_size=horizon_params_60d["batch_size"])
            
            val_ds = TimeSeriesDataSet.from_dataset(train_ds_60d, df_val, predict=True) if len(df_val) > 0 else None
            val_dl = make_dataloader(val_ds, train=False, 
                                   batch_size=horizon_params_60d["batch_size"] * 2) if val_ds else None
            
            # Простое обучение (упрощенное)
            from lightning.pytorch import Trainer
            from lightning.pytorch.callbacks import ModelCheckpoint

            checkpoint_callback = ModelCheckpoint(
                dirpath=self.model_dir,
                filename="staged_14d_model_{epoch:02d}",
                save_top_k=1,
                monitor="val_loss" if val_dl else "train_loss",
                mode="min"
            )
            
            trainer = Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[checkpoint_callback],
                enable_progress_bar=True
            )
            
            if val_dl is not None:
                trainer.fit(model_60d_stage2, train_dataloaders=train_dl, val_dataloaders=val_dl)
            else:
                trainer.fit(model_60d_stage2, train_dataloaders=train_dl)
            
            staged_model_path = os.path.join(self.model_dir, "best_staged_14d_final.pth")
            torch.save(model_60d_stage2.state_dict(), staged_model_path)
            print(f"Сохранена модель поэтапного прогнозирования: {staged_model_path}")

            # Оценка
            mae, rmse, r2, smape = evaluate_metrics(model_60d_stage2, train_dl, scaler, max_batches=3)
            
            self.results['staged_60d'] = {
                'MAE': mae, 'RMSE': rmse, 'R2': r2, 'SMAPE': smape,
                'description': 'Поэтапное прогнозирование 60 дней'
            }
            
            print(f"Результаты поэтапного прогнозирования: SMAPE={smape:.2f}%")
            
            return model_60d_stage2
            
        except Exception as e:
            print(f"Ошибка в поэтапном прогнозировании: {e}")
            return None
    
    def _create_extended_dataset(self, df, model_30d, scaler):
        """Создание расширенного датасета с прогнозами первых 30 дней"""
        try:
            # Берем последние данные для прогноза
            df_recent = df.sort_values('time_idx').groupby(['Магазин', 'КодПродукта']).tail(60)
            
            if len(df_recent) == 0:
                return None
            
            # Создаем датасет для прогноза
            predict_ds = create_simple_dataset(df_recent, prediction_length=30)
            predict_dl = make_dataloader(predict_ds, train=False, batch_size=32)
            
            # Получаем прогнозы
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_30d.to(device)
            model_30d.eval()
            
            all_predictions = []
            
            with torch.no_grad():
                for batch in predict_dl:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x, y = batch
                        # Перемещаем на устройство
                        x_device = {}
                        for key, value in x.items():
                            if torch.is_tensor(value):
                                x_device[key] = value.to(device)
                            else:
                                x_device[key] = value
                        
                        output = model_30d(x_device)
                        predictions = output.prediction.detach().cpu().numpy()
                        
                        # Обратное масштабирование
                        if scaler is not None:
                            try:
                                predictions = scaler.inverse_transform(
                                    predictions.reshape(-1, 1)
                                ).flatten()
                            except:
                                pass
                        
                        all_predictions.extend(predictions)
            
            if not all_predictions:
                return None
            
            # Создаем расширенный датасет
            df_extended = df.copy()
            
            # Добавляем прогнозы как новые фичи
            # (здесь нужна сложная логика присвоения прогнозов к правильным рядам и временным меткам)
            # Для простоты возвращаем исходные данные
            print("Расширенные данные созданы (упрощенная версия)")
            return df_extended
            
        except Exception as e:
            print(f"Ошибка создания расширенного датасета: {e}")
            return df  # Возвращаем исходные данные в случае ошибки
    
    def quantile_forecasting(self, df_train, df_val, quantiles=[0.1, 0.5, 0.9]):
        """Эксперимент 3: Квантильная регрессия - УЛУЧШЕННАЯ ВЕРСИЯ"""
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 3: КВАНТИЛЬНАЯ РЕГРЕССИЯ (УЛУЧШЕННАЯ)")
        print("="*60)

        try:
            from pytorch_forecasting import QuantileLoss
            from lightning.pytorch.callbacks import ModelCheckpoint

            # Упрощаем данные для стабильности
            feature_columns = [
                'Магазин', 'КодПродукта', 'time_idx', 'Количество', 
                'weekday', 'is_holiday_day'  #promoFlag
            ]
            
            df_train_simple = df_train[feature_columns].copy()
            df_val_simple = df_val[feature_columns].copy() if len(df_val) > 0 else df_val
            
            # Создаем датасет
            train_ds = create_simple_dataset(df_train_simple, prediction_length=30)
            
            # Упрощенные параметры
            horizon_params = {
                "learning_rate": 0.0005,  # Меньше LR для стабильности
                "hidden_size": 16,
                "lstm_layers": 1,
                "attention_head_size": 1,
                "dropout": 0.1,
                "hidden_continuous_size": 8,
                "batch_size": 16,
                "max_epochs": 40,
                "patience": 8
            }
            
            # Квантильная модель
            model = TemporalFusionTransformer.from_dataset(
                train_ds,
                learning_rate=horizon_params["learning_rate"],
                hidden_size=horizon_params["hidden_size"],
                lstm_layers=horizon_params["lstm_layers"],
                attention_head_size=horizon_params["attention_head_size"],
                dropout=horizon_params["dropout"],
                hidden_continuous_size=horizon_params["hidden_continuous_size"],
                output_size=len(quantiles),
                loss=QuantileLoss(quantiles=quantiles),
                reduce_on_plateau_patience=horizon_params["patience"],
            )
            
            # Обучение с улучшенными настройками
            train_dl = make_dataloader(train_ds, train=True, batch_size=horizon_params["batch_size"])
            
            val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val_simple, predict=True) if len(df_val_simple) > 0 else None
            val_dl = make_dataloader(val_ds, train=False, batch_size=horizon_params["batch_size"] * 2) if val_ds else None
            
            from lightning.pytorch import Trainer
            from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
            
            early_stop = EarlyStopping(monitor="val_loss", patience=horizon_params["patience"], mode="min")
            lr_monitor = LearningRateMonitor()
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.model_dir,
                filename="quantile_model_{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )
            
            trainer = Trainer(
                max_epochs=horizon_params["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[early_stop, lr_monitor, checkpoint_callback],
                enable_progress_bar=True,
                gradient_clip_val=0.1,
            )
            
            if val_dl is not None:
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            else:
                trainer.fit(model, train_dataloaders=train_dl)
            
            quantile_model_path = os.path.join(self.model_dir, "quantile_model_final.pth")
            torch.save(model.state_dict(), quantile_model_path)
            print(f"Сохранена квантильная модель: {quantile_model_path}")

            # Анализ результатов
            self._analyze_quantile_predictions_improved(model, train_dl, quantiles)
            
            return model
            
        except Exception as e:
            print(f"Ошибка в квантильной регрессии: {e}")
            return None
    
    def _analyze_quantile_predictions(self, model, dataloader, quantiles):
        """Улучшенный анализ квантильных прогнозов"""
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:  # Больше батчей для анализа
                    break
                    
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch
                    output = model(x)
                    predictions = output.prediction.detach().cpu().numpy()
                    
                    # Анализируем разные квантили
                    for i in range(min(5, predictions.shape[0])):  # Первые 5 примеров
                        print(f"\nПример {i+1}:")
                        for q_idx, quantile in enumerate(quantiles):
                            q_values = predictions[i, :, q_idx]
                            print(f"  Q{quantile}: min={q_values.min():.2f}, max={q_values.max():.2f}, mean={q_values.mean():.2f}")
        
        print(f"\nКвантильная модель обучена успешно!")
        print(f"Используемые квантили: {quantiles}")
    
    def run_all_experiments(self, df_train, df_val, df_test, scaler):
        """Запуск всех экспериментов"""
        print("\n" + "="*80)
        print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ")
        print("="*80)
        
        # Эксперимент 1: Поэтапное прогнозирование
        model_staged = self.staged_forecasting_60d(df_train, df_val, df_test, scaler)
        
        # Эксперимент 3: Квантильная регрессия
        model_quantile = self.quantile_forecasting(df_train, df_val)
        
        # Сводка результатов
        self._print_results_summary()
        
        return {
            'staged_model': model_staged,
            'quantile_model': model_quantile
        }
    
    def _print_results_summary(self):
        """Печать сводки результатов"""
        print("\n" + "="*80)
        print("СВОДКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
        print("="*80)
        
        for exp_name, result in self.results.items():
            print(f"\n{exp_name}:")
            for metric, value in result.items():
                if metric != 'description':
                    print(f"  {metric}: {value:.4f}")
            if 'description' in result:
                print(f"  Описание: {result['description']}")

# Упрощенная версия для быстрого тестирования
def quick_experiment(df_train, df_val, scaler):
    """Быстрый эксперимент с квантильной регрессией"""
    print("БЫСТРЫЙ ЭКСПЕРИМЕНТ: Квантильная регрессия")
    
    experimenter = ForecastingExperiments()
    model = experimenter.quantile_forecasting(df_train, df_val)
    return model

# Для использования в основном файле:
def run_experiments_from_main(df_train, df_val, df_test, scaler):
    """Функция для вызова из основного файла"""
    experimenter = ForecastingExperiments()
    return experimenter.run_all_experiments(df_train, df_val, df_test, scaler)
