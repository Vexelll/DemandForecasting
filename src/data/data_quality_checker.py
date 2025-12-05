from datetime import datetime
from config.settings import DATA_PATH, MODELS_PATH


class DataQualityChecker:
    """Проверяет, нужно ли обрабатывать данные и переобучать модель"""

    def has_new_data(self):
        """Проверяет, есть ли новые данные"""
        train_path = DATA_PATH / "raw/train.csv"

        if not train_path.exists():
            return False

        # Данные считаются новыми, если обновлялись сегодня
        train_mtime = datetime.fromtimestamp(train_path.stat().st_mtime)
        return train_mtime.date() == datetime.now().date()

    def needs_retraining(self):
        """Проверяет, нужно ли переобучать модель"""
        model_path = MODELS_PATH / "lgbm_final_model.pkl"
        data_path = DATA_PATH / "processed/final_dataset.csv"

        # Если нет модели - нужно обучить
        if not model_path.exists():
            return True

        # Если нет данных - нельзя проверить, лучше переобучить
        if not data_path.exists():
            return True

        try:
            # Простая проверка: если данные новее модели, то переобучаем
            data_mtime = datetime.fromtimestamp(data_path.stat().st_mtime)
            model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)

            return data_mtime > model_mtime

        except Exception:
            # При любой ошибке - переобучаем
            return True

    def check_data_files(self):
        """Быстрая проверка всех нужных файлов"""
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