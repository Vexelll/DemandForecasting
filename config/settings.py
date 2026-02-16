import logging
import warnings
from pathlib import Path

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"

if not (PROJECT_ROOT / "config").is_dir():
    warnings.warn(
        f"Некорректный PROJECT_ROOT: {PROJECT_ROOT}. Проверьте расположение settings.py (ожидается config/settings.py)",
        RuntimeWarning,
        stacklevel=2
    )

logger = logging.getLogger(__name__)

def all_stores_time_split(df, train_time_ratio=0.7):
    """Разделение всех магазинов по времени (70% начальных данных -> train, 30% конечных -> test)"""
    df = df.sort_values(["Year", "Month", "Week", "Store"]).copy()

    # Находим временную точку разделения для всех магазинов
    all_time_points = df[["Year", "Month", "Week"]].drop_duplicates().sort_values(["Year", "Month", "Week"])

    # Определяем точку разделения между train и test
    split_index = int(len(all_time_points) * train_time_ratio)
    split_point = all_time_points.iloc[split_index]
    split_year, split_month, split_week = split_point["Year"], split_point["Month"], split_point["Week"]

    # Создаем маску для train данных (первые 70% временного периода)
    train_mask = (
            (df["Year"] < split_year) |
            ((df["Year"] == split_year) & (df["Month"] < split_month)) |
            ((df["Year"] == split_year) & (df["Month"] == split_month) & (df["Week"] <= split_week))
    )

    # Создаем маску для test данных (последние 30% временного периода)
    test_mask = (
            (df["Year"] > split_year) |
            ((df["Year"] == split_year) & (df["Month"] > split_month)) |
            ((df["Year"] == split_year) & (df["Month"] == split_month) & (df["Week"] > split_week))
    )

    # Финал
    X_train, y_train = df[train_mask].drop("Sales", axis=1), df[train_mask]["Sales"]
    X_test, y_test = df[test_mask].drop("Sales", axis=1), df[test_mask]["Sales"]

    first_point = all_time_points.iloc[0]
    last_point = all_time_points.iloc[-1]

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    logger.info(f"Магазины: все {df["Store"].nunique()} магазинов в обоих наборах")
    logger.info(f"Диапазон train: {first_point["Year"]}-{first_point["Month"]}-{first_point["Week"]} до {split_year}-{split_month}-{split_week}")
    logger.info(f"Диапазон test: {split_year}-{split_month}-{split_week} до {last_point["Year"]}-{last_point["Month"]}-{last_point["Week"]}")

    return X_train, X_test, y_train, y_test
