import pandas as pd
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager
from src.database.database_manager import DatabaseManager
from config.settings import MODELS_PATH, DATA_PATH

class ETLPipeline:
    # Колонки, исключаемые из финальных признаков
    EXCLUDE_COLUMNS = [
        "Date", "Sales", "Customers", "Open",
        "StateHoliday", "StoreType", "Assortment",
        "PromoInterval", "PromoSequence"
    ]

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.logger = logging.getLogger(__name__)

        if model_path is None:
            model_path = MODELS_PATH / "lgbm_final_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.history_manager = SalesHistoryManager()
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

        # Инициализация БД
        self.db = DatabaseManager.create_database_manager()

        self.logger.info(f"ETL-пайплайн инициализирован. Модель: {model_path}")

    def extract(self, new_data_path: Path, stores_data_path: Path) -> pd.DataFrame:
        """Извлечение и валидация исходных данных"""
        self.logger.info("Извлечение данных...")

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

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Преобразование данных: очистка, feature engineering, обновление истории"""
        self.logger.info("Преобразование данных...")

        # Очистка данных
        cleaned_data = self.preprocessor.clean_data(data)
        self.logger.info(f"Данные очищены: {cleaned_data.shape[0]} записей")

        # Обновление исторических данных
        self.logger.info("Обновление исторических данных...")
        self.history_manager.update_history(cleaned_data)


        stats = self.history_manager.get_history_stats()
        self.logger.info(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

        # Feature Engineering
        df = cleaned_data.copy()

        # Базовые преобразования
        df = self.feature_engineer.handle_nan_values(df)
        df = self.feature_engineer.create_temporal_features(df)
        df = self.feature_engineer.create_promo_features(df)
        df = self.feature_engineer.create_holiday_features(df)
        df = self.feature_engineer.create_store_features(df)
        df = self.feature_engineer.create_competition_features(df)

        # Добавление лаговых признаков из истории
        self.logger.info("Добавление лаговых признаков...")
        lag_df = self.history_manager.calculate_lags_batch(
            df["Store"].values,
            df["Date"].values,
            lag_days=[1, 7, 14, 28]
        )

        # Объединение результатов
        df = pd.merge(df, lag_df, on=["Store", "Date"], how="left")

        # Заполнение пропусков в лаговых признаков
        df = self.feature_engineer.fill_lag_missing_values(df)

        lag_columns = [col for col in df.columns if "Lag" in col or "Rolling" in col]
        remaining_na = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Лаговые признаки добавлены. Остаток NaN: {remaining_na}")

        # Определение финальных признаков
        feature_columns = [col for col in df.columns if col not in self.EXCLUDE_COLUMNS]

        # Валидация совпадения признаков с моделью
        self._validate_feature_columns(feature_columns)

        final_data = df[feature_columns + ["Sales"]]

        self.logger.info(f"Преобразование завершено: {final_data.shape[1]} признаков")
        self.cleaned_data = cleaned_data
        self.processed_data = final_data

        return final_data, feature_columns

    def _validate_feature_columns(self, feature_columns: List[str]) -> None:
        """Проверка совпадения признаков с ожидаемыми моделью"""
        if not hasattr(self.model, "feature_name_"):
            self.logger.debug("Модель не содержит информации о признаках, пропуск валидации")
            return

        model_features = set(self.model.feature_name_)
        pipeline_features = set(feature_columns)

        # Признаки, которые ожидает модель, но нет в пайплайне
        missing_features = model_features - pipeline_features
        if missing_features:
            self.logger.warning(f"Признаки отсутствуют в данных пайплайна (будут NaN): {sorted(missing_features)}")

        # Лишнии признаки, которые модель не ожидает
        extra_features = pipeline_features - model_features
        if extra_features:
            self.logger.debug(f"Дополнительные признаки (будут проигнорированы моделью): {sorted(extra_features)}")

    def store_predictions(self, predictions: np.ndarray, output_path: Path) -> pd.DataFrame:
        """Load-фаза ETL: сохранение прогнозов в хранилище данных (БД/файлы)"""
        self.logger.info("Сохранение прогнозов...")

        if self.cleaned_data is None:
            raise ValueError("Очищенные данные не найдены. Сначала выполните transform().")

        if len(predictions) != len(self.cleaned_data):
            raise ValueError(f"Несоответствие размеров: predictions={len(predictions)}, cleaned_data={len(self.cleaned_data)}")

        # Создание DateFrame c результатами
        results_df = pd.DataFrame({
            "Store": self.cleaned_data["Store"],
            "Date": self.cleaned_data["Date"],
            "PredictedSales": predictions,
            "ActualSales": self.cleaned_data["Sales"]
        })

        # Добавление метрик качества
        results_df["AbsoluteError"] = np.abs(results_df["ActualSales"] - results_df["PredictedSales"])
        results_df["PercentageError"] = np.where(results_df["ActualSales"] > 0, (results_df["AbsoluteError"] / results_df["ActualSales"]) * 100, 0.0)

        try:
            # Создание директории, если не существует
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)

            # Статистика прогнозов
            total_predictions = len(results_df)
            avg_error = results_df["AbsoluteError"].mean()
            mape = results_df["PercentageError"].mean()
            median_error = results_df["AbsoluteError"].median()

            self.logger.info(f"Прогнозы сохранены: {output_path}")
            self.logger.info(f"Статистика: {total_predictions:,} прогнозов, {results_df['Store'].nunique()} магазинов, диапазон дат: {results_df['Date'].min()} - {results_df['Date'].max()}")
            self.logger.info(f"Качество: MAE={avg_error:,.2f} €, MedAE={median_error:,.2f} €, MAPE={mape:.2f}%")

            # Сохранение прогнозов в БД
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
        """Запуск полного ETL-пайплайна"""
        self.logger.info("Запуск ETL-пайплайна прогнозирования спроса")

        try:
            # Extract
            raw_data = self.extract(new_data_path, store_data_path)

            # transform
            processed_data, features = self.transform(raw_data)

            # Проверка наличия данных для прогнозирования
            if len(processed_data) == 0:
                raise ValueError("Нет данных для прогнозирования после преобразований")

            # Predict
            self.logger.info("Генерация прогнозов...")
            predictions = self.model.predict(processed_data[features])

            # Load
            results = self.store_predictions(predictions, output_path)

            self.logger.info("ETL-пайплайн успешно завершен")

            return results

        except Exception as e:
            self.logger.error(f"Критическая ошибка в ETL-пайплайне: {e}")
            raise

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Тестирование ETL-пайплайна...")

    pipeline = ETLPipeline()
    results = pipeline.run_pipeline(
        DATA_PATH / "raw/test.csv",
        DATA_PATH / "raw/store.csv",
        DATA_PATH / "outputs/predictions.csv"
    )

    logger.info("Тестирование завершено успешно")
