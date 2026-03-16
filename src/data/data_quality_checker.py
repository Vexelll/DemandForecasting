import logging
from datetime import datetime
from typing import Any

from config.settings import MODELS_PATH, get_model_config, get_monitoring_config, resolve_data_path


class DataQualityChecker:
    """Проверки для DAG: свежие ли данные, нужно ли переобучать"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def has_new_data(self) -> bool:
        """mtime train.csv < threshold_hours -> данные свежие"""
        train_path = resolve_data_path("raw", "train")

        if not train_path.exists():
            self.logger.warning(f"Файл данных не найден: {train_path}")
            return False

        # Порог свежести из config.yaml -> monitoring.data_freshness_threshold_hours
        threshold_hours = get_monitoring_config().get("data_freshness_threshold_hours", 24)
        train_mtime = datetime.fromtimestamp(train_path.stat().st_mtime)
        is_new = (datetime.now() - train_mtime) < timedelta(hours=threshold_hours)

        self.logger.info(f"Проверка новых данных: mtime={train_mtime.isoformat()}, порог={threshold_hours}ч, is_new={is_new}")

        return is_new

    def needs_retraining(self) -> bool:
        """Данные новее модели -> переобучаем"""
        model_filename = get_model_config().get("model_filename", "lgbm_final_model.pkl")
        model_path = MODELS_PATH / model_filename
        data_path = resolve_data_path("processed", "final_dataset")

        # Нет модели - нужно обучить
        if not model_path.exists():
            self.logger.info("Модель не найдена - требуется обучение")
            return True

        # Нет данных - нельзя проверить, лучше переобучить
        if not data_path.exists():
            self.logger.info("Данные не найдены - переобучение для безопасности")
            return True

        try:
            # Данные новее модели -> переобучаем
            data_mtime = datetime.fromtimestamp(data_path.stat().st_mtime)
            model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)

            needs = data_mtime > model_mtime
            self.logger.info(f"Проверка переобучения: data_mtime={data_mtime.isoformat()}, model_mtime={model_mtime.isoformat()}, needs_retraining={needs}")

            return needs

        except Exception as e:
            # При любой ошибке - переобучаем
            self.logger.error(f"Ошибка проверки переобучения: {e}. Переобучаем")
            return True

    def check_data_files(self) -> dict[str, dict[str, Any]]:
        """Проверяет train.csv, store.csv, модель, predictions - exists + size + mtime"""
        files_to_check = {
            "train.csv": resolve_data_path("raw", "train"),
            "store.csv": resolve_data_path("raw", "store"),
            "model.pkl": MODELS_PATH / get_model_config().get("model_filename", "lgbm_final_model.pkl"),
            "predictions.csv": resolve_data_path("outputs", "predictions")
        }

        result = {}
        for name, path in files_to_check.items():
            exists = path.exists()
            stat = path.stat() if exists else None
            result[name] = {
                "exists": exists,
                "size": stat.st_size if stat else 0,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None
            }

        return result
