import logging
from datetime import datetime
from typing import Dict, Any
from config.settings import DATA_PATH, MODELS_PATH


class DataQualityChecker:
    """Проверяет наличие новых данных и необходимость переобучения модели"""
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def has_new_data(self) -> bool:
        """Проверяет, есть ли новые данные для обработки"""
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
        """Проверяет, нужно ли переобучать модель"""
        model_path = MODELS_PATH / "lgbm_final_model.pkl"
        data_path = DATA_PATH / "processed/final_dataset.csv"

        # Если нет модели - нужно обучить
        if not model_path.exists():
            self.logger.info("Модель не найдена - требуется обучение")
            return True

        # Если нет данных - нельзя проверить, лучше переобучить
        if not data_path.exists():
            self.logger.info("Данные не найдена - переобучение для безопасности")
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
        """Быстрая проверка наличия и размера всех ключевых файлов"""
        files_to_check = {
            "train.csv": DATA_PATH / "raw/train.csv",
            "store.csv": DATA_PATH / "raw/store.csv",
            "model.pkl": MODELS_PATH / "lgbm_final_model.pkl",
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
