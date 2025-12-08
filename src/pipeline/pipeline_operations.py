import pandas as pd
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager
from src.models.lgmb_model import LGBMModel
from src.pipeline.etl_pipeline import ETLPipeline
from config.settings import DATA_PATH


class PipelineOperations:
    """Выполняет основные операции пайплайна"""

    def __init__(self) -> None:
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.history_manager = SalesHistoryManager()

    def preprocess_data(self) -> bool:
        """Очищает данные"""
        try:
            data = self.preprocessor.load_and_merge_data(
                DATA_PATH / "raw/train.csv",
                DATA_PATH / "raw/store.csv"
            )
            cleaned = self.preprocessor.clean_data(data)

            # Сохраняем
            cleaned_path = DATA_PATH / "processed/cleaned_data.csv"
            self.preprocessor.save_processed_data(cleaned, cleaned_path)

            return True
        except Exception as e:
            print(f"Ошибка предобработки данных: {e}")
            return False

    def create_features(self) -> bool:
        """Создает фичи"""
        try:
            cleaned_data = pd.read_csv(
                DATA_PATH / "processed/cleaned_data.csv",
                parse_dates=["Date"]
            )

            final_data, features = self.feature_engineer.prepare_final_dataset(cleaned_data)

            # Сохраняем
            final_data.to_csv(DATA_PATH / "processed/final_dataset.csv", index=False)

            return True
        except Exception as e:
            print(f"Ошибка создания признаков: {e}")
            return False

    def train_model(self) -> bool:
        """Обучает модель"""
        try:
            final_data = pd.read_csv(DATA_PATH / "processed/final_dataset.csv")

            if len(final_data) == 0:
                return False

            # Простое обучение без сложных проверок
            model = LGBMModel()
            metrics, _ = model.run_complete_training()

            return metrics is not None
        except Exception as e:
            print(f"Ошибка обучения модели: {e}")
            return False

    def make_predictions(self) -> bool:
        """Делает прогнозы"""
        try:
            etl = ETLPipeline()
            results = etl.run_pipeline(
                DATA_PATH / "raw/train.csv",
                DATA_PATH / "raw/store.csv",
                DATA_PATH / "outputs/predictions.csv"
            )

            return len(results) > 0
        except Exception as e:
            print(f"Ошибка генерации прогнозов: {e}")
            return False

    def validate_predictions(self) -> bool:
        """Проверяет прогнозы"""
        try:
            predictions = pd.read_csv(DATA_PATH / "outputs/predictions.csv")

            # Простые проверки
            checks = {
                "has_data": len(predictions) > 0,
                "no_negative": (predictions["PredictedSales"] >= 0).all() if "PredictedSales" in predictions.columns else True,
                "no_nan": predictions["PredictedSales"].notna().all() if "PredictedSales" in predictions.columns else True
            }

            return all(checks.values())
        except Exception as e:
            print(f"Ошибка валидации прогнозов: {e}")
            return False

    def update_sales_history(self) -> bool:
        """Обновляет историю продаж"""
        try:
            cleaned_data = pd.read_csv(
                DATA_PATH / "processed/cleaned_data.csv",
                parse_dates=["Date"]
            )

            if len(cleaned_data) == 0:
                print("Ошибка: очищенные данные пусты")
                return False

            # Обновляем историю
            self.history_manager.update_history(cleaned_data)

            # Логируем статистику
            stats = self.history_manager.get_history_stats()

            print(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

            return True
        except Exception as e:
            print(f"Ошибка обновления истории: {e}")
            return False