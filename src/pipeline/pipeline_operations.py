import logging
import pandas as pd
from datetime import datetime
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager
from src.database.database_manager import DatabaseManager
from src.models.lgbm_model import LGBMModel
from src.pipeline.etl_pipeline import ETLPipeline
from config.settings import DATA_PATH


class PipelineOperations:
    """Выполняет основные операции пайплайна прогнозирования"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.history_manager = SalesHistoryManager()

        # Инициализация БД для логирования запусков
        self.db = DatabaseManager.create_database_manager()
        self.current_run_id = DatabaseManager.generate_run_id()
        self._run_started_at = datetime.now()

    def _log_to_db(self, dag_name: str, status: str, records_processed: int = 0, error_message: str = None, **metrics) -> None:
        """Логирование результата операции в БД"""
        if self.db is None:
            return

        try:
            duration = (datetime.now() - self._run_started_at).total_seconds()
            self.db.log_pipeline_run(
                run_id=self.current_run_id,
                dag_name=dag_name,
                status=status,
                started_at=self._run_started_at.isoformat(),
                finished_at=datetime.now().isoformat(),
                duration_seconds=duration,
                records_processed=records_processed,
                mape=metrics.get("mape"),
                rmse=metrics.get("rmse"),
                mae=metrics.get("mae"),
                r2_score=metrics.get("r2_score"),
                error_message=error_message
            )
        except Exception as e:
            self.logger.warning(f"Ошибка логирования в БД: {e}")

    def preprocess_data(self) -> bool:
        """Загрузка, объединение и очистка данных"""
        try:
            data = self.preprocessor.load_and_merge_data(
                DATA_PATH / "raw/train.csv",
                DATA_PATH / "raw/store.csv"
            )
            cleaned = self.preprocessor.clean_data(data)

            # Сохраняем очищенные данные
            cleaned_path = DATA_PATH / "processed/cleaned_data.csv"
            self.preprocessor.save_processed_data(cleaned, cleaned_path)

            self.logger.info(f"Предобработка завершена: {len(cleaned)} записей -> {cleaned_path}")

            self._log_to_db("preprocess_data", "success", records_processed=len(cleaned))
            return True

        except Exception as e:
            self.logger.error(f"Ошибка предобработки данных: {e}")
            self._log_to_db("preprocess_data", "failure", error_message=str(e))
            return False

    def create_features(self) -> bool:
        """Создание признаков из очищенных данных"""
        try:
            cleaned_data = pd.read_csv(
                DATA_PATH / "processed/cleaned_data.csv",
                parse_dates=["Date"]
            )

            final_data, features = self.feature_engineer.prepare_final_dataset(cleaned_data)

            # Сохраняем финальный датасет
            output_path = DATA_PATH / "processed/final_dataset.csv"
            final_data.to_csv(output_path, index=False)

            self.logger.info(f"Признаки созданы: {len(features)} признаков, {len(final_data)} записей -> {output_path}")

            self._log_to_db("create_features", "success", records_processed=len(final_data))
            return True

        except Exception as e:
            self.logger.error(f"Ошибка создания признаков: {e}")
            self._log_to_db("create_features", "failure", error_message=str(e))
            return False

    def train_model(self) -> bool:
        """Обучение модели LightGBM"""
        try:
            final_data = pd.read_csv(DATA_PATH / "processed/final_dataset.csv")

            if len(final_data) == 0:
                self.logger.error("Финальный датасет пуст - обучение невозможно")
                return False

            model = LGBMModel()
            metrics, _ = model.run_complete_training()

            if metrics is not None:
                self.logger.info(f"Модель обучена. MAPE: {metrics.get("MAPE", "N/A")}")

                # Сохранение метрик в БД
                if self.db is not None:
                    try:
                        self.db.save_model_metrics(
                            metrics=metrics,
                            n_features=len(final_data.columns) - 1,
                            n_train_samples=int(len(final_data) * 0.7),
                            n_test_samples=int(len(final_data) * 0.3),
                            best_params=model.best_params
                        )
                    except Exception as e:
                        self.logger.warning(f"Ошибка сохранения метрик в БД: {e}")

                self._log_to_db(
                    "train_model", "success",
                    mape=metrics.get("MAPE"),
                    rmse=metrics.get("RMSE"),
                    mae=metrics.get("MAE"),
                    r2_score=metrics.get("R2")
                )
                return True
            else:
                self.logger.error("Обучение модели не вернуло метрики")
                self._log_to_db("train_model", "failure", error_message="Метрики не получены")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            self._log_to_db("train_model", "failure", error_message=str(e))
            return False

    def make_predictions(self) -> bool:
        """Генерация прогнозов через ETL-пайплайн"""
        try:
            etl = ETLPipeline()
            results = etl.run_pipeline(
                DATA_PATH / "raw/test.csv",
                DATA_PATH / "raw/store.csv",
                DATA_PATH / "outputs/predictions.csv"
            )

            self.logger.info(f"Прогнозы сгенерированы: {len(results)} записей")

            self._log_to_db("make_predictions", "success", records_processed=len(results))
            return len(results) > 0

        except Exception as e:
            self.logger.error(f"Ошибка генерации прогнозов: {e}")
            self._log_to_db("make_predictions", "failure", error_message=str(e))
            return False

    def validate_predictions(self) -> bool:
        """Валидация прогнозов: наличие данных, отсутствие NaN и отрицательных значений"""
        try:
            predictions = pd.read_csv(DATA_PATH / "outputs/predictions.csv")

            # Простые проверки
            checks = {
                "has_data": len(predictions) > 0,
                "no_negative": (predictions["PredictedSales"] >= 0).all() if "PredictedSales" in predictions.columns else True,
                "no_nan": predictions["PredictedSales"].notna().all() if "PredictedSales" in predictions.columns else True
            }

            all_passed = all(checks.values())
            failed_checks = [name for name, passed in checks.items() if not passed]

            if all_passed:
                self.logger.info(f"Валидация пройдена: {len(predictions)} прогнозов")
            else:
                self.logger.warning(f"Валидация не пройдена: {failed_checks}")

            return all_passed

        except Exception as e:
            self.logger.error(f"Ошибка валидации прогнозов: {e}")
            return False

    def update_sales_history(self) -> bool:
        """Обновление исторической базы продаж"""
        try:
            cleaned_data = pd.read_csv(
                DATA_PATH / "processed/cleaned_data.csv",
                parse_dates=["Date"]
            )

            if len(cleaned_data) == 0:
                self.logger.error("Ошибка: очищенные данные пусты")
                return False

            # Обновляем историю
            self.history_manager.update_history(cleaned_data)

            # Логируем статистику
            stats = self.history_manager.get_history_stats()
            self.logger.info(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

            return True

        except Exception as e:
            self.logger.error(f"Ошибка обновления истории: {e}")
            return False
