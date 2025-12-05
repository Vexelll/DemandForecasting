# Система Прогнозирования Спроса для Розничной Сети (В процессе разработки)

---
## Обзор проекта
Cистема машинного обучения для прогнозирования ежедневных продаж в аптеках Rossmann с автоматизированным ETL-пайплайном и интерактивным дашбордом.

---
## Архитектура системы
```
demand_forecasting/
├── airflow/                          
│   └── dags/
│       ├── demand_forecasting_dag.py          # Основной DAG для прогнозирования
│       └── monitoring/
│           └── dag_monitor.py                 # Мониторинг выполнения DAG
├── config/                           
│   └── settings.py                            # Настройки путей и параметров
├── data/                             
│   ├── raw/                          # Сырые данные 
│   │   ├── train.csv                          # Исторические данные продаж
│   │   ├── test.csv                           # Тестовые данные
│   │   └── store.csv                          # Справочник магазинов
│   ├── processed/                    # Обработанные данные
│   │   ├── cleaned_data.csv                   # Очищенные данные
│   │   ├── final_dataset.csv                  # Данные с признаками
│   │   └── sales_history.pkl                  # Исторические данные
│   └── outputs/                      # Результаты 
│       └── predictions.csv                    # Результаты прогнозов
├── models/                           
│   └── lgbm_final_model.pkl                   # Обученная модель LightGBM
│
├── notebooks/                        # Анализ и эксперименты
│   ├── 01_initial_analysis.ipynb              # Первичный анализ данных
│   ├── 02_eda.ipynb                           # Разведочный анализ данных
│   └── 03_model_comparison.ipynb              # Сравнение ML-моделей               
├── reports/...                         # Отчеты и визуализации
│
├── scripts/                          
│   ├── setup_wsl.py                           # Настройка WSL окружения
│   └── run_airflow.py                         # Запуск Apache Airflow
├── src/                              
│   ├── data/                         
│   │   ├── preprocessing.py                   # Предобработка данных
│   │   ├── data_quality_checker.py            # Проверка качества данных
│   │   └── history_manager.py                 # Управление историческими данными
│   ├── features/                     
│   │   └── feature_engineering.py             # Генерация признаков
│   ├── models/                       
│   │   ├── base_model.py                      # Метрики и отчеты
│   │   ├── baseline_models.py                 # Базовые модели
│   │   └── lgbm_model.py                      # LightGBM модель
│   ├── pipeline/                     
│   │   ├── etl_pipeline.py                    # ETL пайплайн
│   │   └── pipeline_operations.py             # Операции пайплайна
│   └── visualization/                
│       ├── dashboard.py                       # Основной дашборд
│       └── components/                        
│           ├── layout.py                      # Компоновка интерфейса
│           ├── charts.py                      # Графики и визуализации
│           ├── controls.py                    # Элементы управления
│           └── metrics.py                     # Отображение метрик
│       └── assets/                            
│           ├── style.css                      # Стилизация
│           └── custom.js                      # Кастомный JavaScript
├── tests/                            
│   ├── test_pipeline.py                       # Тесты пайплайна
│   └── test_dag_monitoring.py                 # Тесты мониторинга DAG
├── requirements.txt                           # Зависимости Python
└── README.md                                  # Документация проекта
```
---
## Технологический стек
- **Оркестрация**: Apache Airflow
- **ML-фреймворк**: LightGBM + Optuna
- **Визуализация**: Dash/Plotly
- **Обработка данных**: Pandas, NumPy
- **Тестирование**: pytest, unittest
- **Окружение**: WSL2

---
## Установка

# Клонировать репозиторий
```git clone https://github.com/Vexelll/DemandForecasting.git``` -> ```cd demand_forecasting```

# Создать и активировать виртуальное окружение
```python -m venv venv```(**Для Windows**: ```venv\Scripts\activate```)

# Установить зависимости
```pip install -r requirements.txt```

# Настроить WSL окружение (Windows)
```python scripts/setup_wsl.py``` (p.s. нужно установить WSL2, если не установлен)

# Быстрый старт
1. **Инициализировать Airflow**: ```python scripts/run_airflow.py```
2. **Открыть Airflow UI**: ```http://localhost:8080```
3. **Запустить дашборд**: ```python src/visualization/dashboard.py``` -> ```http://localhost:8050```



