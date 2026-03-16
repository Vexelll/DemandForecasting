import logging
from datetime import datetime

import pandas as pd

from config.settings import resolve_data_path
from src.data.history_manager import SalesHistoryManager
from src.data.preprocessing import DataPreprocessor
from src.database.database_manager import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.models.lgbm_model import LGBMModel
from src.pipeline.etl_pipeline import ETLPipeline


class PipelineOperations:
    """Операции Airflow DAG - каждый метод = одна задача"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.history_manager = SalesHistoryManager()

        self.db = DatabaseManager.create_database_manager()
        self.current_run_id = DatabaseManager.generate_run_id()
        self._run_started_at = datetime.now()

    def _reset_timer(self) -> None:
        """datetime.now() -> _run_started_at перед каждой задачей"""
        self._run_started_at = datetime.now()

    def _log_to_db(self, dag_name: str, status: str, records_processed: int = 0, error_message: str = None, **metrics) -> None:
        """Пишет в pipeline_runs: run_id, статус, метрики, длительность"""
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
        """train.csv + store.csv -> cleaned_data.csv"""
        self._reset_timer()
        try:
            data = self.preprocessor.load_and_merge_data(
                resolve_data_path("raw", "train"),
                resolve_data_path("raw", "store")
            )
            cleaned = self.preprocessor.clean_data(data)

            cleaned_path = resolve_data_path("processed", "cleaned")
            self.preprocessor.save_processed_data(cleaned, cleaned_path)

            self.logger.info(f"Очистка: {len(cleaned)} записей -> {cleaned_path}")

            self._log_to_db("preprocess_data", "success", records_processed=len(cleaned))
            return True

        except Exception as e:
            self.logger.error(f"Ошибка предобработки данных: {e}")
            self._log_to_db("preprocess_data", "failure", error_message=str(e))
            return False

    def create_features(self) -> bool:
        """cleaned_data.csv -> final_dataset.csv c 60+ признаками"""
        self._reset_timer()
        try:
            cleaned_data = pd.read_csv(
                resolve_data_path("processed", "cleaned"),
                parse_dates=["Date"]
            )

            final_data, features = self.feature_engineer.prepare_final_dataset(cleaned_data)

            output_path = resolve_data_path("processed", "final_dataset")
            final_data.to_csv(output_path, index=False)

            self.logger.info(f"Признаки готовы: {len(features)} шт, {len(final_data)} строк -> {output_path}")

            self._log_to_db("create_features", "success", records_processed=len(final_data))
            return True

        except Exception as e:
            self.logger.error(f"Ошибка создания признаков: {e}")
            self._log_to_db("create_features", "failure", error_message=str(e))
            return False

    def train_model(self) -> bool:
        """Optuna optimize -> train final -> save model + metrics"""
        self._reset_timer()
        try:
            dataset_path = resolve_data_path("processed", "final_dataset")
            if not dataset_path.exists():
                self.logger.error("Финальный датасет не найден - обучение невозможно")
                self._log_to_db("train_model", "failure", error_message="Датасет не найден")
                return False

            model = LGBMModel()
            metrics, _, y_test = model.run_complete_training()

            if metrics is not None:
                mape_value = metrics.get("MAPE", "N/A")
                self.logger.info(f"Модель обучена. MAPE: {mape_value}")

                if self.db is not None:
                    try:
                        self.db.save_model_metrics(
                            metrics=metrics,
                            n_features=len(model.feature_names),
                            n_test_samples=len(y_test),
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
                self.logger.error(f"Ошибка обучения модели: {e}")
                self._log_to_db("train_model", "failure", error_message="Метрики не получены")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            self._log_to_db("train_model", "failure", error_message=str(e))
            return False

    def make_predictions(self) -> bool:
        """Поднимает ETLPipeline, прогоняет predict, пишет в outputs/"""
        self._reset_timer()
        try:
            etl = ETLPipeline()
            results = etl.run_pipeline(
                resolve_data_path("raw", "test"),
                resolve_data_path("raw", "store"),
                resolve_data_path("outputs", "predictions")
            )

            self.logger.info(f"Прогнозы: {len(results)} записей")

            self._log_to_db("make_predictions", "success", records_processed=len(results))
            return len(results) > 0

        except Exception as e:
            self.logger.error(f"Ошибка генерации прогнозов: {e}")
            self._log_to_db("make_predictions", "failure", error_message=str(e))
            return False

    def validate_predictions(self) -> bool:
        """Проверяет predictions.csv: есть колонка, не пустой, нет NaN, нет отрицательных"""
        self._reset_timer()
        try:
            predictions = pd.read_csv(resolve_data_path("outputs", "predictions"))

            # Нет PredictedSales - остальные проверки бессмысленны
            if "PredictedSales" not in predictions.columns:
                self.logger.error(f"Колонка PredictedSales отсутствует в predictions.csv")
                self._log_to_db("validate_predictions", "failure", error_message="PredictedSales not found")
                return False

            checks = {
                "has_data": len(predictions) > 0,
                "no_negative": (predictions["PredictedSales"] >= 0).all(),
                "no_nan": predictions["PredictedSales"].notna().all()
            }

            all_passed = all(checks.values())
            failed_checks = [name for name, passed in checks.items() if not passed]

            if all_passed:
                self.logger.info(f"Валидация ок: {len(predictions)} прогнозов")
                self._log_to_db("validate_predictions", "success", records_processed=len(predictions))
            else:
                self.logger.warning(f"Валидация не пройдена: {failed_checks}")
                self._log_to_db("validate_predictions", "failure", error_message=f"Проверка не пройдена: {failed_checks}")

            return all_passed

        except Exception as e:
            self.logger.error(f"Ошибка валидации прогнозов: {e}")
            self._log_to_db("validate_predictions", "failure", error_message=str(e))
            return False

    def update_sales_history(self) -> bool:
        """Добавляет cleaned_data в историю (БД + pickle)"""
        self._reset_timer()
        try:
            cleaned_data = pd.read_csv(
                resolve_data_path("processed", "cleaned"),
                parse_dates=["Date"]
            )

            if len(cleaned_data) == 0:
                self.logger.error("Ошибка: очищенные данные пусты")
                self._log_to_db("update_sales_history", "failure", error_message="Пустой датасет")
                return False

            self.history_manager.update_history(cleaned_data)

            stats = self.history_manager.get_history_stats()
            self.logger.info(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

            self._log_to_db("update_sales_history", "success", records_processed=len(cleaned_data))
            return True

        except Exception as e:
            self.logger.error(f"Ошибка обновления истории: {e}")
            self._log_to_db("update_sales_history", "failure", error_message=str(e))
            return False
