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

def all_stores_time_split(df, train_time_ratio=0.8):
    """Разделение всех магазинов по времени (70% начальных данных -> train, 30% конечных -> test)"""
    if not 0.0 < train_time_ratio < 1.0:
        raise ValueError(f"train_time_ratio должен быть в диапазоне (0, 1), получено: {train_time_ratio}")

    df = df.sort_values(["Year", "Week", "Store"]).copy()

    # Уникальные временные периоды (Year, Week) - монотонно возрастают
    all_time_points = (
        df[["Year", "Week"]]
        .drop_duplicates()
        .sort_values(["Year", "Week"])
        .reset_index(drop=True)
    )

    # Определяем точку разделения между train и test
    split_index = int(len(all_time_points) * train_time_ratio)
    split_point = all_time_points.iloc[split_index]
    split_year = split_point["Year"]
    split_week = split_point["Week"]

    # Создаем маску для train данных - все периоды до split_point включительно
    train_mask = (
        (df["Year"] < split_year) |
        ((df["Year"] == split_year) & (df["Week"] <= split_week))
    )

    # Создаем маску для test данных - все периоды строго после split_point
    test_mask = (
        (df["Year"] > split_year) |
        ((df["Year"] == split_year) & (df["Week"] > split_week))
    )

    # Финал
    X_train, y_train = df[train_mask].drop("Sales", axis=1), df[train_mask]["Sales"]
    X_test, y_test = df[test_mask].drop("Sales", axis=1), df[test_mask]["Sales"]

    first_point = all_time_points.iloc[0]
    last_point = all_time_points.iloc[-1]

    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    logger.info(f"Магазины: все {df["Store"].nunique()} магазинов в обоих наборах")
    logger.info(f"Диапазон train: {first_point["Year"]}-W{first_point["Week"]} до {split_year}-W{split_week}")
    logger.info(f"Диапазон test: {split_year}-W{split_week} до {last_point["Year"]}-W{last_point["Week"]}")
    logger.info(f"Размеры: train={len(X_train)}, test={len(X_test)} (ratio={len(X_train) / (len(X_train) + len(X_test)):.2f})")

    return X_train, X_test, y_train, y_test
