import logging
from datetime import datetime
from typing import Dict, Any
from config.settings import DATA_PATH, MODELS_PATH, get_model_config


class DataQualityChecker:
    """Проверки для DAG: свежие ли данные, нужно ли переобучать"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def has_new_data(self) -> bool:
        """Данные считаются новыми, если mtime = сегодня (простой алгоритм)"""
        train_path = DATA_PATH / "raw/train.csv"

        if not train_path.exists():
            self.logger.warning(f"Файл данных не найден: {train_path}")
            return False

        # Данные считаются новыми, если обновлялись сегодня
        train_mtime = datetime.fromtimestamp(train_path.stat().st_mtime)
        is_new = train_mtime.date() == datetime.now().date()

        self.logger.info(f"Проверка новых данных: mtime={train_mtime.isoformat()}, is_new={is_new}")

        return is_new

    def needs_retraining(self) -> bool:
        """Если модели нет, или она старше 7 дней -> переобучаем"""
        model_filename = get_model_config().get("model_filename", "lgbm_final_model.pkl")
        model_path = MODELS_PATH / model_filename
        data_path = DATA_PATH / "processed/final_dataset.csv"

        # Если нет модели - нужно обучить
        if not model_path.exists():
            self.logger.info("Модель не найдена - требуется обучение")
            return True

        # Если нет данных - нельзя проверить, лучше переобучить
        if not data_path.exists():
            self.logger.info("Данные не найдены - переобучение для безопасности")
            return True

        try:
            # Простая проверка: если данные новее модели, то переобучаем
            data_mtime = datetime.fromtimestamp(data_path.stat().st_mtime)
            model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)

            needs = data_mtime > model_mtime
            self.logger.info(f"Проверка переобучения: data_mtime={data_mtime.isoformat()}, model_mtime={model_mtime.isoformat()}, needs_retraining={needs}")

            return needs

        except Exception as e:
            # При любой ошибке - переобучаем
            self.logger.error(f"Ошибка проверки переобучения: {e}. Переобучаем")
            return True

    def check_data_files(self) -> Dict[str, Dict[str, Any]]:
        """Проверяет, что train.csv, store.csv, модель на месте"""
        files_to_check = {
            "train.csv": DATA_PATH / "raw/train.csv",
            "store.csv": DATA_PATH / "raw/store.csv",
            "model.pkl": MODELS_PATH / get_model_config().get("model_filename", "lgbm_final_model.pkl"),
            "predictions.csv": DATA_PATH / "outputs/predictions.csv"
        }

        result = {}
        for name, path in files_to_check.items():
            exists = path.exists()
            result[name] = {
                "exists": exists,
                "size": path.stat().st_size if exists else 0
            }

        return result
