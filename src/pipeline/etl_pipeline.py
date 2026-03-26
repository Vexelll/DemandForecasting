import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager
from src.database.database_manager import DatabaseManager
from config.settings import MODELS_PATH, resolve_data_path, setup_logging, get_model_config, get_feature_config

class ETLPipeline:

    EXCLUDE_COLUMNS = FeatureEngineer.EXCLUDE_COLUMNS

    def __init__(self, model_path: Path | None = None) -> None:
        self.logger = logging.getLogger(__name__)

        if model_path is None:
            model_filename = get_model_config().get("model_filename", "lgbm_final_model.pkl")
            model_path = MODELS_PATH / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.history_manager = SalesHistoryManager()
        self.cleaned_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None

        self.db = DatabaseManager.create_database_manager()

        self.logger.info(f"ETL-пайплайн инициализирован. Модель: {model_path}")

    def extract(self, new_data_path: Path, stores_data_path: Path) -> pd.DataFrame:
        """Загрузка train.csv + store.csv, merge по Store"""
        self.logger.info("Загрузка данных...")

        if not new_data_path.exists():
            raise FileNotFoundError(f"Файл с новыми данными не найден: {new_data_path}")
        if not stores_data_path.exists():
            raise FileNotFoundError(f"Файл с информацией о магазинах не найден: {stores_data_path}")

        try:
            new_data = self.preprocessor.load_and_merge_data(new_data_path, stores_data_path)
            self.logger.info(f"Данные извлечены: {new_data.shape[0]} записей, {new_data.shape[1]} признаков")
            return new_data
        except Exception as e:
            self.logger.error(f"Ошибка извлечения данных: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Очистка -> признаки -> лаги из истории -> финальный датасет"""
        self.logger.info("Очистка + feature engineering...")

        cleaned_data = self.preprocessor.clean_data(data)
        self.logger.info(f"Данные очищены: {cleaned_data.shape[0]} записей")

        self.logger.info("Обновляю историю продаж...")
        self.history_manager.update_history(cleaned_data)

        stats = self.history_manager.get_history_stats()
        self.logger.info(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

        df = cleaned_data.copy()
        df = self.feature_engineer.handle_nan_values(df)
        df = self.feature_engineer.create_temporal_features(df)
        df = self.feature_engineer.create_promo_features(df)
        df = self.feature_engineer.create_holiday_features(df)
        df = self.feature_engineer.create_store_features(df)
        df = self.feature_engineer.create_competition_features(df)

        # Лаги считаются из истории, а не из текущего df - чтобы не было утечки
        self.logger.info("Считаю лаги из истории...")
        lag_df = self.history_manager.calculate_lags_batch(
            df["Store"].values,
            df["Date"].values,
            lag_days=get_feature_config().get("lag_days", [1, 7, 14, 28])
        )

        df = pd.merge(df, lag_df, on=["Store", "Date"], how="left")

        df = self.feature_engineer.fill_lag_missing_values(df)

        lag_columns = [col for col in df.columns if "Lag" in col or "Rolling" in col]
        remaining_na = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Лаговые признаки добавлены. Остаток NaN: {remaining_na}")

        # Определение финальных признаков
        feature_columns = [col for col in df.columns if col not in self.EXCLUDE_COLUMNS]
        feature_columns = self._validate_feature_columns(df, feature_columns)

        final_data = df[feature_columns + ["Sales"]]

        self.logger.info(f"Преобразование завершено: {final_data.shape[1]} признаков")
        self.cleaned_data = cleaned_data
        self.processed_data = final_data

        return final_data, feature_columns

    def _validate_feature_columns(self, df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
        """Сверяет колонки с model.feature_name_, недостающие -> NaN, лишние -> убираем"""
        if not hasattr(self.model, "feature_name_"):
            self.logger.debug("Модель не содержит информации о признаках, пропуск валидации")
            return feature_columns

        model_features = self.model.feature_name_
        pipeline_features = set(feature_columns)

        missing = [f for f in model_features if f not in pipeline_features]
        if missing:
            self.logger.warning(f"Признаки отсутствуют в данных (заполнены NaN): {sorted(missing)}")
            for col in missing:
                df[col] = np.nan

        extra = [f for f in feature_columns if f not in set(model_features)]
        if extra:
            self.logger.debug(f"Лишние признаки (убраны): {sorted(extra)}")

        return model_features

    def store_predictions(self, predictions: np.ndarray, output_path: Path) -> pd.DataFrame:
        """Сохраняет прогнозы в CSV + дублирует в БД, если доступна"""
        self.logger.info("Записываю прогнозы...")

        if self.cleaned_data is None:
            raise ValueError("Очищенные данные не найдены. Сначала выполните transform()")

        if len(predictions) != len(self.cleaned_data):
            raise ValueError(f"Несоответствие размеров: predictions={len(predictions)}, cleaned_data={len(self.cleaned_data)}")

        results_df = pd.DataFrame({
            "Store": self.cleaned_data["Store"],
            "Date": self.cleaned_data["Date"],
            "PredictedSales": predictions,
            "ActualSales": self.cleaned_data["Sales"]
        })

        # Построчные ошибки для CSV, агрегат - в лог
        results_df["AbsoluteError"] = np.abs(results_df["ActualSales"] - results_df["PredictedSales"])
        results_df["PercentageError"] = np.where(results_df["ActualSales"] > 0, (results_df["AbsoluteError"] / results_df["ActualSales"]) * 100, 0.0)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)

            total_predictions = len(results_df)
            avg_error = results_df["AbsoluteError"].mean()
            mape = results_df["PercentageError"].mean()
            median_error = results_df["AbsoluteError"].median()

            self.logger.info(f"Прогнозы сохранены: {output_path}")
            self.logger.info(f"Статистика: {total_predictions:,} прогнозов, {results_df["Store"].nunique()} магазинов, диапазон дат: {results_df["Date"].min()} - {results_df["Date"].max()}")
            self.logger.info(f"Качество: MAE={avg_error:,.2f} €, MedAE={median_error:,.2f} €, MAPE={mape:.2f}%")

            if self.db is not None:
                try:
                    run_id = DatabaseManager.generate_run_id()
                    self.db.save_predictions(results_df, run_id)
                    self.logger.info(f"Прогнозы сохранены в БД (run_id={run_id})")
                except Exception as e:
                    self.logger.warning(f"Ошибка сохранения прогнозов в БД: {e}")
            return results_df

        except Exception as e:
            self.logger.error(f"Ошибка сохранения прогнозов: {e}")
            raise

    def run_pipeline(self, new_data_path: Path, store_data_path: Path, output_path: Path) -> pd.DataFrame:
        """Extract -> Transform -> Predict -> Load, весь цикл"""
        self.logger.info("Запуск ETL-пайплайна")

        try:
            # Extract
            raw_data = self.extract(new_data_path, store_data_path)

            # Transform
            processed_data, features = self.transform(raw_data)

            if len(processed_data) == 0:
                raise ValueError("Нет данных для прогнозирования после преобразований")

            # Predict
            self.logger.info("Прогнозирование...")
            predictions = np.expm1(self.model.predict(processed_data[features]))

            # Load
            results = self.store_predictions(predictions, output_path)

            self.logger.info("ETL завершен")

            return results

        except Exception as e:
            self.logger.error(f"Критическая ошибка в ETL-пайплайне: {e}")
            raise

if __name__ == "__main__":
    # Настройка логирования
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Тестирование ETL-пайплайна...")

    pipeline = ETLPipeline()
    results = pipeline.run_pipeline(
        resolve_data_path("raw", "test"),
        resolve_data_path("raw", "store"),
        resolve_data_path("outputs", "predictions")
    )

    logger.info("Тестирование завершено успешно")
