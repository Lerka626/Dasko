import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

class SalesAnalyzer:
    """
    Класс для комплексного анализа данных продаж
    """
    
    def __init__(self, db_path: str = "only_alco_final.db"):
        """
        Инициализация анализатора
        
        Args:
            db_path: путь к базе данных SQLite
        """
        self.db_path = db_path
        self.df = None
        self.features_df = None
        
    
    def load_data(self, 
                  start_date: str = None, 
                  end_date: str = None,
                  shops: List[str] = None,
                  categories: List[str] = None) -> pd.DataFrame:
        """
        Загрузка данных из БД с фильтрацией
        
        Args:
            start_date: начальная дата (YYYY-MM-DD)
            end_date: конечная дата (YYYY-MM-DD)
            shops: список магазинов
            categories: список категорий
            
        Returns:
            DataFrame с данными
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        
        # Базовый запрос
        query = """
        SELECT 
            Дата,
            Магазин,
            НоменклатураКод,
            Категория,
            GAP,
            Количество,
            promoFlag,
            is_holiday_day
        FROM sales_data
        WHERE 1=1
        """
        
        params = []
        
        # Добавляем фильтры
        if start_date:
            query += " AND Дата >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND Дата <= ?"
            params.append(end_date)
            
        if shops:
            placeholders = ','.join(['?'] * len(shops))
            query += f" AND Магазин IN ({placeholders})"
            params.extend(shops)
            
        if categories:
            placeholders = ','.join(['?'] * len(categories))
            query += f" AND Категория IN ({placeholders})"
            params.extend(categories)
        
        query += " ORDER BY Дата, Магазин, НоменклатураКод"
        
        self.df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Преобразуем типы
        self.df['Дата'] = pd.to_datetime(self.df['Дата'])
        self.df['Количество'] = pd.to_numeric(self.df['Количество'], errors='coerce').fillna(0)
        self.df['promoFlag'] = self.df['promoFlag'].astype(int)
        self.df['is_holiday_day'] = self.df['is_holiday_day'].astype(int)
        
        print(f"Загружено {len(self.df):,} записей")
        print(f"   Период: {self.df['Дата'].min().date()} - {self.df['Дата'].max().date()}")
        print(f"   Магазинов: {self.df['Магазин'].nunique()}")
        print(f"   Товаров: {self.df['НоменклатураКод'].nunique()}")
        print(f"   Категорий: {self.df['Категория'].nunique()}")
        
        return self.df
    
    
    def analyze_distributions(self, output_dir: str = "reports") -> Dict:
        """
        Анализ распределений продаж
        
        Args:
            output_dir: директория для сохранения графиков
            
        Returns:
            Словарь с метриками распределений
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nАНАЛИЗ РАСПРЕДЕЛЕНИЙ ПРОДАЖ")
        
        # 1. Распределение дневных продаж
        daily_sales = self.df.groupby('Дата')['Количество'].sum()
        
        plt.figure(figsize=(15, 10))
        
        # 1.1. Общее распределение
        plt.subplot(2, 2, 1)
        plt.hist(daily_sales, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(daily_sales.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее: {daily_sales.mean():.1f}')
        plt.axvline(daily_sales.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Медиана: {daily_sales.median():.1f}')
        plt.xlabel('Продажи в день')
        plt.ylabel('Частота')
        plt.title('Распределение дневных продаж')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 1.2. Boxplot по дням недели
        self.df['ДеньНедели'] = self.df['Дата'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sales_by_day = self.df.groupby(['Дата', 'ДеньНедели'])['Количество'].sum().reset_index()
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=sales_by_day, x='ДеньНедели', y='Количество', 
                   order=day_order, palette='Set2')
        plt.title('Распределение продаж по дням недели')
        plt.xlabel('День недели')
        plt.ylabel('Продажи')
        plt.xticks(rotation=45)
        
        # 1.3. Распределение по магазинам
        plt.subplot(2, 2, 3)
        shop_sales = self.df.groupby('Магазин')['Количество'].sum().sort_values(ascending=False)
        shop_sales.head(20).plot(kind='bar', color='coral', alpha=0.7)
        plt.title('Топ-20 магазинов по объему продаж')
        plt.xlabel('Магазин')
        plt.ylabel('Общие продажи')
        plt.xticks(rotation=90)
        
        # 1.4. Распределение по категориям
        plt.subplot(2, 2, 4)
        cat_sales = self.df.groupby('Категория')['Количество'].sum().sort_values(ascending=False)
        cat_sales.head(15).plot(kind='bar', color='lightgreen', alpha=0.7)
        plt.title('Топ-15 категорий по объему продаж')
        plt.xlabel('Категория')
        plt.ylabel('Общие продажи')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Метрики распределений
        metrics = {
            'total_sales': float(self.df['Количество'].sum()),
            'avg_daily_sales': float(daily_sales.mean()),
            'median_daily_sales': float(daily_sales.median()),
            'std_daily_sales': float(daily_sales.std()),
            'cv_daily_sales': float(daily_sales.std() / daily_sales.mean()),  # Коэффициент вариации
            'top_3_shops': shop_sales.head(3).to_dict(),
            'top_3_categories': cat_sales.head(3).to_dict(),
            'daily_stats': {
                'min': float(daily_sales.min()),
                'max': float(daily_sales.max()),
                'q25': float(daily_sales.quantile(0.25)),
                'q75': float(daily_sales.quantile(0.75))
            }
        }
        
        return metrics
    
    def analyze_seasonality(self, output_dir: str = "reports") -> Dict:
        """
        Анализ сезонности продаж
        
        Args:
            output_dir: директория для сохранения графиков
            
        Returns:
            Словарь с метриками сезонности
        """
        
        # Агрегация по разным временным интервалам
        self.df['Год'] = self.df['Дата'].dt.year
        self.df['Месяц'] = self.df['Дата'].dt.month
        self.df['Квартал'] = self.df['Дата'].dt.quarter
        self.df['НеделяГода'] = self.df['Дата'].dt.isocalendar().week
        
        # Месячные продажи
        monthly_sales = self.df.groupby(['Год', 'Месяц'])['Количество'].sum().reset_index()
        monthly_sales['Период'] = monthly_sales['Год'].astype(str) + '-' + monthly_sales['Месяц'].astype(str).str.zfill(2)
        
        # Квартальные продажи
        quarterly_sales = self.df.groupby(['Год', 'Квартал'])['Количество'].sum().reset_index()
        quarterly_sales['Период'] = quarterly_sales['Год'].astype(str) + '-Q' + quarterly_sales['Квартал'].astype(str)
        
        plt.figure(figsize=(15, 10))
        
        # 1. Тренд по месяцам
        plt.subplot(2, 2, 1)
        plt.plot(monthly_sales['Период'], monthly_sales['Количество'], 
                marker='o', linewidth=2, markersize=6, color='steelblue')
        plt.fill_between(monthly_sales['Период'], monthly_sales['Количество'], 
                        alpha=0.3, color='steelblue')
        plt.title('Месячные продажи (тренд)')
        plt.xlabel('Месяц')
        plt.ylabel('Продажи')
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3)
        
        # 2. Сезонность по месяцам (усреднение по годам)
        monthly_pattern = self.df.groupby('Месяц')['Количество'].mean()
        
        plt.subplot(2, 2, 2)
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        plt.bar(range(1, 13), monthly_pattern.values, 
               color=plt.cm.Set3(np.arange(12)/12), alpha=0.8)
        plt.axhline(monthly_pattern.mean(), color='red', linestyle='--', 
                   label=f'Среднее: {monthly_pattern.mean():.1f}')
        plt.title('Сезонность по месяцам (средние продажи)')
        plt.xlabel('Месяц')
        plt.ylabel('Средние продажи')
        plt.xticks(range(1, 13), months, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Продажи по дням недели
        day_sales = self.df.groupby('ДеньНедели')['Количество'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday', 'Sunday'
        ])
        days_ru = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        
        plt.subplot(2, 2, 3)
        plt.bar(days_ru, day_sales.values, color='lightcoral', alpha=0.7)
        plt.axhline(day_sales.mean(), color='red', linestyle='--',
                   label=f'Среднее: {day_sales.mean():.1f}')
        plt.title('Средние продажи по дням недели')
        plt.xlabel('День недели')
        plt.ylabel('Средние продажи')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Heatmap: год × месяц
        pivot_monthly = monthly_sales.pivot(index='Год', columns='Месяц', values='Количество')
        
        plt.subplot(2, 2, 4)
        sns.heatmap(pivot_monthly, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Продажи'})
        plt.title('Тепловая карта: продажи по годам и месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Год')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/seasonality.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Метрики сезонности
        seasonality_metrics = {
            'monthly_seasonality': monthly_pattern.to_dict(),
            'daily_seasonality': day_sales.to_dict(),
            'peak_month': int(monthly_pattern.idxmax()),
            'peak_month_sales': float(monthly_pattern.max()),
            'low_month': int(monthly_pattern.idxmin()),
            'low_month_sales': float(monthly_pattern.min()),
            'seasonality_strength': float(monthly_pattern.std() / monthly_pattern.mean()),
            'quarterly_growth': self._calculate_growth_rates(quarterly_sales, 'Квартал')
        }
        
        return seasonality_metrics
    
    def analyze_sparse_sku(self, threshold_days: int = 30, output_dir: str = "reports") -> Dict:
        """
        Анализ sparse (редких) SKU
        
        Args:
            threshold_days: порог дней для определения sparse SKU
            output_dir: директория для сохранения отчетов
            
        Returns:
            Словарь с метриками sparse SKU
        """
        
        # Анализ активности товаров
        sku_activity = self.df.groupby('НоменклатураКод').agg({
            'Дата': ['count', 'nunique'],
            'Количество': ['sum', 'mean', 'std']
        })
        
        # Выравниваем мультииндекс
        sku_activity.columns = ['total_records', 'active_days', 'total_sales', 
                               'avg_sales', 'std_sales']
        
        # Классификация SKU
        sku_activity['sales_frequency'] = sku_activity['active_days'] / self.df['Дата'].nunique()
        sku_activity['is_sparse'] = sku_activity['active_days'] <= threshold_days
        sku_activity['cv_sales'] = sku_activity['std_sales'] / sku_activity['avg_sales']
        sku_activity['cv_sales'] = sku_activity['cv_sales'].replace([np.inf, -np.inf], np.nan)
        
        # Распределение sparse SKU по категориям
        sparse_by_category = self.df[self.df['НоменклатураКод'].isin(
            sku_activity[sku_activity['is_sparse']].index
        )].groupby('Категория')['НоменклатураКод'].nunique()
        
        plt.figure(figsize=(15, 10))
        
        # 1. Распределение sparse vs non-sparse
        plt.subplot(2, 2, 1)
        sparse_counts = sku_activity['is_sparse'].value_counts()
        labels = ['Не sparse', 'Sparse']
        colors = ['lightgreen', 'lightcoral']
        plt.pie(sparse_counts, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0.05))
        plt.title(f'Распределение SKU (sparse ≤ {threshold_days} дней)')
        
        # 2. Sparse SKU по категориям (топ-10)
        plt.subplot(2, 2, 2)
        sparse_by_category.sort_values(ascending=False).head(10).plot(
            kind='bar', color='salmon', alpha=0.7)
        plt.title('Топ-10 категорий по количеству sparse SKU')
        plt.xlabel('Категория')
        plt.ylabel('Количество sparse SKU')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Распределение частоты продаж
        plt.subplot(2, 2, 3)
        plt.hist(sku_activity['sales_frequency'], bins=50, 
                color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(sku_activity['sales_frequency'].median(), color='red', 
                   linestyle='--', label='Медиана')
        plt.axvline(sku_activity['sales_frequency'].mean(), color='green', 
                   linestyle='--', label='Среднее')
        plt.title('Распределение частоты продаж SKU')
        plt.xlabel('Частота продаж (дни активности / всего дней)')
        plt.ylabel('Количество SKU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Топ sparse SKU по объему продаж
        top_sparse = sku_activity[sku_activity['is_sparse']].nlargest(10, 'total_sales')
        
        plt.subplot(2, 2, 4)
        plt.barh(range(len(top_sparse)), top_sparse['total_sales'], 
                color='lightblue', alpha=0.7)
        plt.yticks(range(len(top_sparse)), top_sparse.index)
        plt.title('Топ-10 sparse SKU по объему продаж')
        plt.xlabel('Общие продажи')
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/sparse_sku.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Метрики sparse SKU
        sparse_metrics = {
            'total_sku': len(sku_activity),
            'sparse_sku_count': int(sku_activity['is_sparse'].sum()),
            'sparse_percentage': float(sku_activity['is_sparse'].mean() * 100),
            'avg_active_days': float(sku_activity['active_days'].mean()),
            'median_active_days': float(sku_activity['active_days'].median()),
            'sku_by_activity_level': self._categorize_sku_activity(sku_activity),
            'sparse_by_category': sparse_by_category.to_dict(),
            'top_sparse_sku': top_sparse[['total_sales', 'active_days']].to_dict('index')
        }
        
        return sparse_metrics
    
    def analyze_outliers(self, method: str = 'iqr', multiplier: float = 1.5, 
                        output_dir: str = "reports") -> Dict:
        """
        Анализ выбросов и пиков продаж
        
        Args:
            method: метод обнаружения выбросов ('iqr' или 'zscore')
            multiplier: множитель для IQR метода
            output_dir: директория для сохранения графиков
            
        Returns:
            Словарь с информацией о выбросах
        """
        
        # Дневные продажи
        daily_sales = self.df.groupby('Дата')['Количество'].sum().reset_index()
        
        # Обнаружение выбросов
        if method == 'iqr':
            Q1 = daily_sales['Количество'].quantile(0.25)
            Q3 = daily_sales['Количество'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = daily_sales[(daily_sales['Количество'] < lower_bound) | 
                                  (daily_sales['Количество'] > upper_bound)]
        else:  # zscore
            from scipy import stats
            z_scores = np.abs(stats.zscore(daily_sales['Количество'].fillna(0)))
            outliers = daily_sales[z_scores > multiplier]
        
        # Анализ пиков (топ дней)
        top_days = daily_sales.nlargest(10, 'Количество')
        
        plt.figure(figsize=(15, 10))
        
        # 1. Временной ряд с выбросами
        plt.subplot(2, 2, 1)
        plt.plot(daily_sales['Дата'], daily_sales['Количество'], 
                alpha=0.6, linewidth=1, color='steelblue', label='Продажи')
        
        # Выделяем выбросы
        if not outliers.empty:
            plt.scatter(outliers['Дата'], outliers['Количество'], 
                       color='red', s=50, zorder=5, label='Выбросы')
        
        # Средняя линия
        plt.axhline(daily_sales['Количество'].mean(), color='green', 
                   linestyle='--', alpha=0.7, label='Среднее')
        
        # Границы выбросов (если метод IQR)
        if method == 'iqr':
            plt.axhline(upper_bound, color='orange', linestyle=':', 
                       alpha=0.5, label='Верхняя граница')
            plt.axhline(lower_bound, color='orange', linestyle=':', 
                       alpha=0.5, label='Нижняя граница')
        
        plt.title('Временной ряд продаж с выбросами')
        plt.xlabel('Дата')
        plt.ylabel('Продажи')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Boxplot продаж
        plt.subplot(2, 2, 2)
        plt.boxplot(daily_sales['Количество'], vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        plt.title('Boxplot дневных продаж')
        plt.xlabel('Продажи')
        plt.grid(True, alpha=0.3)
        
        # 3. Топ дней по продажам
        plt.subplot(2, 2, 3)
        bars = plt.barh(range(len(top_days)), top_days['Количество'], 
                       color=plt.cm.Reds(np.linspace(0.3, 0.9, len(top_days))))
        plt.yticks(range(len(top_days)), top_days['Дата'].dt.strftime('%Y-%m-%d'))
        plt.title('Топ-10 дней по продажам')
        plt.xlabel('Продажи')
        
        # Добавляем значения на барчарт
        for i, (bar, value) in enumerate(zip(bars, top_days['Количество'])):
            plt.text(value, i, f'{value:,.0f}', va='center', ha='left', fontsize=9)
        
        # 4. Анализ причин выбросов
        plt.subplot(2, 2, 4)
        if not outliers.empty:
            # Анализ дней недели для выбросов
            outlier_days = outliers['Дата'].dt.day_name().value_counts()
            outlier_days = outlier_days.reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                'Friday', 'Saturday', 'Sunday'
            ]).fillna(0)
            
            plt.bar(range(len(outlier_days)), outlier_days.values, 
                   color='lightcoral', alpha=0.7)
            plt.xticks(range(len(outlier_days)), ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
            plt.title('Выбросы по дням недели')
            plt.xlabel('День недели')
            plt.ylabel('Количество выбросов')
        else:
            plt.text(0.5, 0.5, 'Выбросы не обнаружены', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/outliers.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Метрики выбросов
        outlier_metrics = {
            'total_days': len(daily_sales),
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(daily_sales) * 100,
            'avg_sales': float(daily_sales['Количество'].mean()),
            'median_sales': float(daily_sales['Количество'].median()),
            'max_sales': float(daily_sales['Количество'].max()),
            'min_sales': float(daily_sales['Количество'].min()),
            'top_outliers': outliers.nlargest(5, 'Количество').to_dict('records'),
            'outlier_dates': outliers['Дата'].dt.strftime('%Y-%m-%d').tolist(),
            'detection_method': method,
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)} if method == 'iqr' else {}
        }
        
        return outlier_metrics
    
    
    def create_features(self, target_sku: str = None, lookback_days: int = 30) -> pd.DataFrame:
        """
        Создание признаков для прогнозирования
        
        Args:
            target_sku: целевой SKU (если None - агрегируем все)
            lookback_days: количество дней для лаговых признаков
            
        Returns:
            DataFrame с признаками
        """
        
        # Фильтруем данные если указан конкретный SKU
        if target_sku:
            df_sku = self.df[self.df['НоменклатураКод'] == target_sku].copy()
            print(f"Анализ SKU: {target_sku}")
        else:
            df_sku = self.df.copy()
        
        # Агрегация по дням
        daily_data = df_sku.groupby('Дата').agg({
            'Количество': 'sum',
            'promoFlag': 'max',
            'is_holiday_day': 'max',
            'Магазин': 'nunique',
            'НоменклатураКод': 'nunique'
        }).reset_index()
        
        daily_data.columns = ['Дата', 'sales', 'promo', 'holiday', 
                            'active_shops', 'active_sku']
        
        # Создаем временные признаки
        daily_data = self._create_time_features(daily_data)
        
        # Лаговые признаки
        daily_data = self._create_lag_features(daily_data, lookback_days)
        
        # Статистические признаки
        daily_data = self._create_statistical_features(daily_data, lookback_days)
        
        # Бизнес-признаки
        daily_data = self._create_business_features(daily_data)
        
        # Целевая переменная (продажи на следующий день)
        daily_data['target'] = daily_data['sales'].shift(-1)
        
        # Удаляем строки с NaN
        daily_data = daily_data.dropna()
        
        self.features_df = daily_data
        
        return daily_data
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков"""
        df['year'] = df['Дата'].dt.year
        df['month'] = df['Дата'].dt.month
        df['quarter'] = df['Дата'].dt.quarter
        df['week'] = df['Дата'].dt.isocalendar().week
        df['day'] = df['Дата'].dt.day
        df['dayofweek'] = df['Дата'].dt.dayofweek
        df['dayofyear'] = df['Дата'].dt.dayofyear
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['Дата'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Дата'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Дата'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Дата'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['Дата'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['Дата'].dt.is_year_end.astype(int)
        
        # Праздничные признаки (расширенные)
        df['days_to_holiday'] = self._days_to_nearest_holiday(df['Дата'])
        df['is_pre_holiday'] = (df['days_to_holiday'] == 1).astype(int)
        df['is_post_holiday'] = (df['days_to_holiday'] == -1).astype(int)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        """Создание лаговых признаков"""
        # Лаги продаж
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
        # Лаги по дням недели (сезонные лаги)
        for i in range(1, 5):
            df[f'sales_lag_week_{i}'] = df['sales'].shift(i * 7)
        
        # Скользящие статистики
        for window in [3, 7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window=window, min_periods=1).mean().shift(1)
            df[f'sales_rolling_std_{window}'] = df['sales'].rolling(window=window, min_periods=1).std().shift(1)
            df[f'sales_rolling_min_{window}'] = df['sales'].rolling(window=window, min_periods=1).min().shift(1)
            df[f'sales_rolling_max_{window}'] = df['sales'].rolling(window=window, min_periods=1).max().shift(1)
            df[f'sales_rolling_median_{window}'] = df['sales'].rolling(window=window, min_periods=1).median().shift(1)
        
        # Экспоненциальное сглаживание
        df['sales_ewm_alpha_0.3'] = df['sales'].ewm(alpha=0.3).mean().shift(1)
        df['sales_ewm_alpha_0.7'] = df['sales'].ewm(alpha=0.7).mean().shift(1)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        """Создание статистических признаков"""
        # Процентные изменения
        df['sales_pct_change_1'] = df['sales'].pct_change(1)
        df['sales_pct_change_7'] = df['sales'].pct_change(7)
        df['sales_pct_change_30'] = df['sales'].pct_change(30)
        
        # Статистики за последние N дней
        for days in [7, 14, 30]:
            df[f'sales_mean_last_{days}'] = df['sales'].rolling(days).mean().shift(1)
            df[f'sales_std_last_{days}'] = df['sales'].rolling(days).std().shift(1)
            df[f'sales_cv_last_{days}'] = df[f'sales_std_last_{days}'] / df[f'sales_mean_last_{days}']
            df[f'sales_trend_last_{days}'] = self._calculate_trend(df['sales'], days)
        
        # Статистики по дням недели
        for day in range(7):
            day_mask = df['dayofweek'] == day
            day_sales = df.loc[day_mask, 'sales']
            if len(day_sales) > 1:
                day_mean = day_sales.expanding().mean().shift(1)
                df.loc[day_mask, f'sales_mean_dow_{day}'] = day_mean
                df.loc[day_mask, f'sales_std_dow_{day}'] = day_sales.expanding().std().shift(1)
        
        return df
    
    def _create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание бизнес-ориентированных признаков"""
        # Взаимодействие признаков
        df['promo_effect'] = df['promo'] * df['sales_rolling_mean_7']
        df['holiday_effect'] = df['holiday'] * df['sales_rolling_mean_7']
        df['weekend_promo'] = df['is_weekend'] * df['promo']
        
        # Признаки активности
        df['sales_per_shop'] = df['sales'] / df['active_shops'].replace(0, 1)
        df['sales_per_sku'] = df['sales'] / df['active_sku'].replace(0, 1)
        
        # Отклонение от тренда
        df['sales_deviation_from_mean_7'] = df['sales'] - df['sales_rolling_mean_7']
        df['sales_deviation_from_mean_30'] = df['sales'] - df['sales_rolling_mean_30']
        
        # Признаки волатильности
        df['sales_volatility_ratio'] = df['sales_rolling_std_7'] / df['sales_rolling_mean_7']
        df['sales_momentum'] = df['sales'] - df['sales_lag_7']
        
        # Календарные бизнес-признаки
        df['is_payday'] = ((df['day'] >= 25) & (df['day'] <= 31)) | ((df['day'] >= 5) & (df['day'] <= 10))
        df['is_season_start'] = df['month'].isin([1, 4, 9])  # Начало сезонов
        df['is_season_end'] = df['month'].isin([3, 6, 12])  # Конец сезонов
        
        return df
    
    
    def _calculate_growth_rates(self, df: pd.DataFrame, period_col: str) -> Dict:
        """Расчет темпов роста"""
        growth = {}
        
        if len(df) > 1:
            # Квартальный/месячный рост
            df = df.sort_values(period_col)
            df['growth'] = df['Количество'].pct_change()
            
            # Средний рост
            growth['avg_growth'] = float(df['growth'].mean())
            growth['median_growth'] = float(df['growth'].median())
            growth['positive_growth_periods'] = int((df['growth'] > 0).sum())
            growth['negative_growth_periods'] = int((df['growth'] < 0).sum())
            
            # Последний период
            if len(df) > 1:
                growth['last_growth'] = float(df['growth'].iloc[-1])
        
        return growth
    
    def _categorize_sku_activity(self, sku_df: pd.DataFrame) -> Dict:
        """Категоризация SKU по уровню активности"""
        categories = {
            'high_frequency': (sku_df['active_days'] >= sku_df['active_days'].quantile(0.75)).sum(),
            'medium_frequency': ((sku_df['active_days'] >= sku_df['active_days'].quantile(0.25)) & 
                               (sku_df['active_days'] < sku_df['active_days'].quantile(0.75))).sum(),
            'low_frequency': (sku_df['active_days'] < sku_df['active_days'].quantile(0.25)).sum(),
            'very_low_frequency': (sku_df['active_days'] <= 7).sum()
        }
        return categories
    
    def _days_to_nearest_holiday(self, dates: pd.Series) -> pd.Series:
        """Расчет дней до ближайшего праздника"""
        from datetime import datetime
        
        # Примерные праздники (можно расширить)
        holidays = [
            '01-01', '01-02', '01-03', '01-04', '01-05', '01-06', '01-07', '01-08',
            '02-23', '03-08', '05-01', '05-09', '06-12', '11-04'
        ]
        
        result = []
        for date in dates:
            date_str = date.strftime('%m-%d')
            min_diff = 365
            
            for holiday in holidays:
                holiday_date = datetime.strptime(f"{date.year}-{holiday}", "%Y-%m-%d")
                diff = (holiday_date - date).days
                if abs(diff) < abs(min_diff):
                    min_diff = diff
            
            result.append(min_diff)
        
        return pd.Series(result, index=dates.index)
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Расчет тренда (наклон линейной регрессии)"""
        from scipy import stats
        
        trend = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series) + 1):
            y = series.iloc[i-window:i].values
            x = np.arange(len(y))
            
            if len(y) > 1 and not np.all(y == y[0]):
                slope, _, _, _, _ = stats.linregress(x, y)
                trend.iloc[i-1] = slope
            else:
                trend.iloc[i-1] = 0
        
        return trend.shift(1)
    
    
    def generate_report(self, output_dir: str = "reports", 
                       start_date: str = None,
                       end_date: str = None) -> Dict:
        """
        Генерация комплексного отчета
        
        Args:
            output_dir: директория для сохранения отчетов
            start_date: начальная дата
            end_date: конечная дата
            
        Returns:
            Словарь со всеми метриками
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем данные
        self.load_data(start_date=start_date, end_date=end_date)
        
        # Выполняем все анализы
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {
                'start_date': self.df['Дата'].min().strftime('%Y-%m-%d'),
                'end_date': self.df['Дата'].max().strftime('%Y-%m-%d'),
                'total_days': self.df['Дата'].nunique()
            },
            'data_summary': {
                'total_records': len(self.df),
                'total_shops': self.df['Магазин'].nunique(),
                'total_sku': self.df['НоменклатураКод'].nunique(),
                'total_categories': self.df['Категория'].nunique(),
                'total_sales': float(self.df['Количество'].sum())
            },
            'distributions': self.analyze_distributions(output_dir),
            'seasonality': self.analyze_seasonality(output_dir),
            'sparse_sku': self.analyze_sparse_sku(output_dir=output_dir),
            'outliers': self.analyze_outliers(output_dir=output_dir)
        }
        
        # Создаем признаки
        report['features'] = {
            'feature_count': len(self.create_features().columns) - 2,
            'feature_categories': {
                'time_features': ['year', 'month', 'dayofweek', 'is_weekend', 'is_holiday'],
                'lag_features': ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7'],
                'statistical_features': ['sales_pct_change', 'sales_std_last_7', 'sales_trend'],
                'business_features': ['promo_effect', 'holiday_effect', 'sales_per_shop']
            }
        }
        
        # Сохраняем отчет
        report_file = f"{output_dir}/sales_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # Создаем текстовый отчет
        self._create_text_report(report, output_dir)
        
        print(f"\nОтчет сохранен в директории: {output_dir}/")
        print(f"   • sales_analysis_report.json - полный отчет")
        print(f"   • sales_analysis_summary.txt - текстовая сводка")
        print(f"   • distributions.png - графики распределений")
        print(f"   • seasonality.png - графики сезонности")
        print(f"   • sparse_sku.png - анализ sparse SKU")
        print(f"   • outliers.png - анализ выбросов")
        
        return report
    

def main():
    """Пример использования модуля анализа"""
    
    # 1. Инициализация анализатора
    analyzer = SalesAnalyzer("only_aclo_final.db")
    
    # 2. Ограничимся годом данных
    analyzer.load_data(
        start_date="2024-06-04",
        end_date="2025-06-04",
        # shops=["Магазин_01", "Магазин_02"],
        # categories=["Вино", "Виски"]
    )
    
    # Распределения
    dist_metrics = analyzer.analyze_distributions("reports")
    
    # Сезонность
    season_metrics = analyzer.analyze_seasonality("reports")
    
    # Sparse SKU
    sparse_metrics = analyzer.analyze_sparse_sku(output_dir="reports")
    
    # Выбросы
    outlier_metrics = analyzer.analyze_outliers(output_dir="reports")
    
    # Создание признаков
    features_df = analyzer.create_features(lookback_days=30)
    
    print("ОТЧЕТ")
    
    full_report = analyzer.generate_report(
        output_dir="reports",
        start_date="2024-01-01",
        end_date="2025-06-04"
    )
    
    if features_df is not None:
        features_df.to_csv("reports/features_dataset.csv", index=False, encoding='utf-8-sig')
        print("Признаки сохранены в: reports/features_dataset.csv")


if __name__ == "__main__":
    main()