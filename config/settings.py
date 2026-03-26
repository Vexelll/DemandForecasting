import logging
import yaml
import warnings
from pathlib import Path
from typing import Any


# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

if not (PROJECT_ROOT / "config").is_dir():
    warnings.warn(
        f"Некорректный PROJECT_ROOT: {PROJECT_ROOT}. Проверьте расположение settings.py (ожидается config/settings.py)",
        RuntimeWarning,
        stacklevel=2
    )

def _load_config() -> dict[str, Any]:
    """Читает config.yaml, возвращает dict или {}, если файла нет"""
    if not CONFIG_PATH.exists():
        warnings.warn(
           f"Файл конфигурации не найден: {CONFIG_PATH}. Используются значения по умолчанию",
           RuntimeWarning,
           stacklevel=2
        )
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        warnings.warn(
            f"Некорректный формат config.yaml: ожидался словарь, получен {type(config).__name__}",
            RuntimeWarning,
            stacklevel=2
        )
        return {}

    return config

# Глобальный объект конфигурации
_config = _load_config()

# Пути проекта
DATA_PATH = PROJECT_ROOT / _config.get("paths", {}).get("data", "data")
MODELS_PATH = PROJECT_ROOT / _config.get("paths", {}).get("models", "models")
REPORTS_PATH = PROJECT_ROOT / _config.get("paths", {}).get("reports", "reports")

# Типизированный доступ к секциям конфигурации
def get_config() -> dict[str, Any]:
    """Копия всего конфига"""
    return _config.copy()

def get_model_config() -> dict[str, Any]:
    """Секция model из config.yaml"""
    return _config.get("model", {})

def get_optimization_config() -> dict[str, Any]:
    """Секция optimization из config.yaml"""
    return _config.get("optimization", {})

def get_feature_config() -> dict[str, Any]:
    """Секция feature_engineering из config.yaml"""
    return _config.get("feature_engineering", {})

def get_pipeline_config() -> dict[str, Any]:
    """Секция pipeline из config.yaml"""
    return _config.get("pipeline", {})

def get_dashboard_config() -> dict[str, Any]:
    """Секция dashboard из config.yaml"""
    return _config.get("dashboard", {})

def get_reporting_config() -> dict[str, Any]:
    """Секция reporting из config.yaml"""
    return _config.get("reporting", {})

def get_logging_config() -> dict[str, Any]:
    """Секция logging из config.yaml"""
    return _config.get("logging", {})

def get_data_files_config() -> dict[str, Any]:
    """Секция data_files из config.yaml"""
    return _config.get("data_files", {})

def get_monitoring_config() -> dict[str, Any]:
    """Секция monitoring из config.yaml"""
    return _config.get("monitoring", {})

def get_database_config() -> dict[str, Any]:
    """Секция database из config.yaml"""
    return _config.get("database", {})


def resolve_data_path(section: str, key: str) -> Path:
    """data_files.raw.train -> DATA_PATH/raw/train.csv"""
    data_files = get_data_files_config()
    relative = data_files.get(section, {}).get(key)
    if relative is None:
        raise KeyError(f"Путь не найден в конфигурации: data_files.{section}.{key}")
    return DATA_PATH / relative

def setup_logging() -> None:
    """basicConfig из параметров config.yaml"""
    log_cfg = get_logging_config()
    level_name = log_cfg.get("level", "INFO")
    log_format = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    date_format = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format=log_format,
        datefmt=date_format
    )

logger = logging.getLogger(__name__)

def all_stores_time_split(df, train_time_ratio=None):
    """Все магазины в обоих наборах, split по ISOYear-Week (дефолт 80/20)"""
    if train_time_ratio is None:
            train_time_ratio = get_model_config().get("train_time_ratio", 0.8)

    if not 0.0 < train_time_ratio < 1.0:
        raise ValueError(f"train_time_ratio должен быть в диапазоне (0, 1), получено: {train_time_ratio}")

    df = df.sort_values(["ISOYear", "Week", "Store"]).copy()

    # Уникальные (ISOYear, Week) пары - по ним делим, чтобы не разрезать неделю пополам
    all_time_points = (
        df[["ISOYear", "Week"]]
        .drop_duplicates()
        .sort_values(["ISOYear", "Week"])
        .reset_index(drop=True)
    )

    split_index = int(len(all_time_points) * train_time_ratio)
    split_point = all_time_points.iloc[split_index]
    split_year = split_point["ISOYear"]
    split_week = split_point["Week"]

    train_mask = (
        (df["ISOYear"] < split_year) |
        ((df["ISOYear"] == split_year) & (df["Week"] <= split_week))
    )

    test_mask = (
        (df["ISOYear"] > split_year) |
        ((df["ISOYear"] == split_year) & (df["Week"] > split_week))
    )

    # Финал
    X_train, y_train = df[train_mask].drop("Sales", axis=1), df[train_mask]["Sales"]
    X_test, y_test = df[test_mask].drop("Sales", axis=1), df[test_mask]["Sales"]

    first_point = all_time_points.iloc[0]
    last_point = all_time_points.iloc[-1]

    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    logger.info(f"Магазины: все {df["Store"].nunique()} магазинов в обоих наборах")
    logger.info(f"Диапазон train: {first_point["ISOYear"]}-W{first_point["Week"]} до {split_year}-W{split_week}")
    logger.info(f"Диапазон test: {split_year}-W{split_week} до {last_point["ISOYear"]}-W{last_point["Week"]}")
    logger.info(f"Размеры: train={len(X_train)}, test={len(X_test)} (ratio={len(X_train) / (len(X_train) + len(X_test)):.2f})")

    return X_train, X_test, y_train, y_test
