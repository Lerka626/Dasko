import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def debug_batch_structure(dataloader):
    """Отладочная функция для изучения структуры батча - ИСПРАВЛЕННАЯ"""
    print("=== ОТЛАДКА СТРУКТУРЫ БАТЧА ===")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:
            break
            
        print(f"Тип батча: {type(batch)}")
        print(f"Длина батча: {len(batch) if isinstance(batch, (list, tuple)) else 'не список'}")
        
        if isinstance(batch, (list, tuple)):
            for i, item in enumerate(batch):
                print(f"  Элемент {i}: тип={type(item)}, форма={item.shape if hasattr(item, 'shape') else 'нет формы'}")
                
                # Если это тензор, покажем его свойства
                if hasattr(item, 'shape'):
                    print(f"    Размеры: {item.shape}, диапазон: {item.min():.3f} - {item.max():.3f}")
        
        # Проверяем специальные атрибуты TFT
        if hasattr(batch, '__dict__'):
            print("Атрибуты батча:", batch.__dict__.keys())
        
        break

def plot_predictions_vs_actual_tft_fixed(model, dataloader, scaler=None, n_examples=3, horizon=30):
    """Визуализация для TFT формата батчей - ИСПРАВЛЕННАЯ"""
    print("=== ВИЗУАЛИЗАЦИЯ ДЛЯ TFT (ИСПРАВЛЕННАЯ) ===")
    
    # ДОБАВЬТЕ ЭТО - перемещение модели на устройство
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            try:
                # Батч - это список, где [0] - encoder данные, [1] - decoder данные
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch
                    
                    # ДОБАВЬТЕ ЭТО - перемещение данных на устройство
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
                    
                    # Получаем прогнозы с исправленными данными
                    output = model(x_device)  # ← используем x_device вместо x
                    prediction = output.prediction.detach().cpu().numpy()
                    
                    # Получаем целевые значения
                    target = y_device[0].detach().cpu().numpy()  # decoder_target
                    
                    print(f"Батч {batch_idx}: prediction shape {prediction.shape}, target shape {target.shape}")
                    
                    # Выравниваем массивы
                    prediction = prediction.reshape(-1)
                    target = target.reshape(-1)
                    
                    # Обратное масштабирование
                    if scaler is not None:
                        try:
                            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                            target = scaler.inverse_transform(target.reshape(-1, 1)).flatten()
                        except Exception as e:
                            print(f"Ошибка масштабирования: {e}")
                    
                    # Обрезаем отрицательные значения
                    prediction = np.maximum(prediction, 0)
                    target = np.maximum(target, 0)
                    
                    all_predictions.extend(prediction)
                    all_actuals.extend(target)
                    
                else:
                    print(f"Батч {batch_idx}: неожиданный формат")
                    
            except Exception as e:
                print(f"Ошибка в батче {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not all_predictions:
        print("Нет данных для визуализации!")
        return None, None
    
    # Конвертируем в numpy arrays
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    
    print(f"Всего точек для визуализации: {len(predictions)}")
    print(f"Диапазон факт: {actuals.min():.2f} - {actuals.max():.2f}")
    print(f"Диапазон прогноз: {predictions.min():.2f} - {predictions.max():.2f}")
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Прогнозы vs Факт (Горизонт {horizon} дней)', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.6, s=20, color='blue')
    max_val = max(actuals.max(), predictions.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Фактические продажи')
    axes[0, 0].set_ylabel('Прогнозируемые продажи')
    axes[0, 0].set_title('Факт vs Прогноз')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Распределение ошибок
    errors = predictions - actuals
    axes[0, 1].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Ошибка прогноза')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение ошибок')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Временной ряд (первые n примеров)
    n_show = min(50, len(predictions))
    time_points = range(n_show)
    
    axes[1, 0].plot(time_points, actuals[:n_show], 'b-', label='Факт', linewidth=2, alpha=0.8)
    axes[1, 0].plot(time_points, predictions[:n_show], 'r--', label='Прогноз', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Временные точки')
    axes[1, 0].set_ylabel('Продажи')
    axes[1, 0].set_title('Примеры временных рядов')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    data_to_plot = [actuals, predictions]
    axes[1, 1].boxplot(data_to_plot, labels=['Факт', 'Прогноз'])
    axes[1, 1].set_ylabel('Продажи')
    axes[1, 1].set_title('Распределение продаж')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Статистика
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    smape_val = 100 * np.mean(2 * np.abs(predictions - actuals) / (np.abs(actuals) + np.abs(predictions) + 1e-8))
    
    stats_text = f"""Статистика:
MAE: {mae:.3f}
RMSE: {rmse:.3f}
SMAPE: {smape_val:.1f}%
Среднее факт: {actuals.mean():.3f}
Среднее прогноз: {predictions.mean():.3f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'tft_predictions_horizon_{horizon}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return predictions, actuals

def quick_visualization_fixed(model, dataloader, scaler=None, horizon=30):
    """Быстрая визуализация - ИСПРАВЛЕННАЯ"""
    print("=== БЫСТРАЯ ВИЗУАЛИЗАЦИЯ (ИСПРАВЛЕННАЯ) ===")
    
    # ДОБАВЬТЕ ЭТО
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        
        try:
            # Батч - это список [x, y]
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch
                
                # ДОБАВЬТЕ ЭТО - перемещение данных
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
                
                # Используем исправленные данные
                output = model(x_device)  # ← используем x_device вместо x
                pred = output.prediction.detach().cpu().numpy().flatten()
                
                # y содержит [decoder_target, ...]
                target = y_device[0].detach().cpu().numpy().flatten()
                
                if scaler:
                    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    target = scaler.inverse_transform(target.reshape(-1, 1)).flatten()
                
                pred = np.maximum(pred, 0)
                target = np.maximum(target, 0)
                
                # Простой график
                plt.figure(figsize=(12, 6))
                plt.plot(target, 'bo-', label='Факт', markersize=4, linewidth=1)
                plt.plot(pred, 'ro--', label='Прогноз', markersize=4, linewidth=1)
                plt.title(f'Прогнозы vs Факт (Горизонт {horizon} дней) - Первый батч')
                plt.xlabel('Временные точки в батче')
                plt.ylabel('Продажи')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'quick_plot_horizon_{horizon}.png', dpi=300, bbox_inches='tight')
                # plt.show()
                
                print(f"Визуализировано {len(pred)} точек")
                print(f"Факт (первые 10): {target[:10]}")
                print(f"Прогноз (первые 10): {pred[:10]}")
                
            else:
                print("Неожиданный формат батча")
                
        except Exception as e:
            print(f"Ошибка быстрой визуализации: {e}")
            import traceback
            traceback.print_exc()

            