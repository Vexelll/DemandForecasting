import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager
from config.settings import MODELS_PATH, DATA_PATH

class ETLPipeline:
    def __init__(self, model_path: Optional[Path] = None) -> None:
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

    def extract(self, new_data_path: Path, stores_data_path: Path) -> pd.DataFrame:
        """Извлечение и валидация исходных данных"""
        print("Извлечение данных...")

        if not new_data_path.exists():
            raise FileNotFoundError(f"Файл с новыми данными не найден: {new_data_path}")
        if not stores_data_path.exists():
            raise FileNotFoundError(f"Файл с информацией о магазинах не найден: {stores_data_path}")

        try:
            new_data = self.preprocessor.load_and_merge_data(new_data_path, stores_data_path)
            print(f"Данные извлечены: {new_data.shape[0]} записей, {new_data.shape[1]} признаков")
            return new_data
        except Exception as e:
            print(f"Ошибка извлечения данных: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Преобразование данных: очистка, feature engineering, обновление истории"""
        print("Преобразование данных...")

        # Очистка данных
        cleaned_data = self.preprocessor.clean_data(data)
        print(f"Данные очищены: {cleaned_data.shape[0]} записей")

        # Обновление исторических данных
        print("Обновление исторических данных...")


        self.history_manager.update_history(cleaned_data)


        stats = self.history_manager.get_history_stats()
        print(f"История обновлена: {stats["total_records"]} записей, {stats["unique_stores"]} магазинов")

        # Feature Engineering
        df = cleaned_data.copy()

        # Базовые преобразования
        df = self.feature_engineer.handle_nan_values(df)
        df = self.feature_engineer.create_temporal_features(df)
        df = self.feature_engineer.create_promo_features(df)
        df = self.feature_engineer.create_holiday_features(df)
        df = self.feature_engineer.create_store_features(df)

        # Добавление лаговых признаков из истории
        print("Добавление лаговых признаков...")
        lag_features = []

        for idx, row in df.iterrows():
            lags = self.history_manager.calculate_lags(
                row["Store"],
                row["Date"],
                lag_days=[1, 7, 14, 28]
            )
            lag_features.append(lags)

        # Объединение лагов с основными признаками
        lag_df = pd.DataFrame(lag_features, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        # Определение финальных признаков
        exclude_columns = [
            "Date", "Sales", "Customers", "Open", "StateHoliday", "StoreType", "Assortment", "PromoInterval"
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        final_data = df[feature_columns + ["Sales"]]

        print(f"Преобразование завершено: {final_data.shape[1]} признаков")
        self.cleaned_data = cleaned_data
        self.processed_data = final_data

        return final_data, feature_columns

    def load_predictions(self, predictions: np.ndarray, output_path: Path) -> pd.DataFrame:
        """Сохранение прогнозов и метаданных"""
        print("Сохранение прогнозов")

        if self.cleaned_data is None:
            raise ValueError("Очищенные данные не найдены. Сначала выполните transform().")

        # Создание DateFrame c результатами
        results_df = pd.DataFrame({
            "Store": self.cleaned_data["Store"],
            "Date": self.cleaned_data["Date"],
            "PredictedSales": predictions,
            "ActualSales": self.cleaned_data["Sales"]
        })

        # Добавление метрик качества
        results_df["AbsoluteError"] = abs(results_df["ActualSales"] - results_df["PredictedSales"])
        results_df["PercentageError"] = results_df["PercentageError"] = (results_df["AbsoluteError"] / self.cleaned_data["Sales"]) * 100

        try:
            results_df.to_csv(output_path, index=False)

            # Дополнительная статистика
            total_predictions = len(results_df)
            avg_errors = results_df["AbsoluteError"].mean()
            mape = results_df["PercentageError"].mean()

            print(f"Прогнозы сохранены: {output_path}")
            print("Статистика прогнозов:")
            print(f"Всего прогнозов: {total_predictions}")
            print(f"Средняя абсолютная ошибка: {avg_errors:.2f} €")
            print(f"MAPE: {mape:.2f}%")
            print(f"Магазинов: {results_df["Store"].nunique()}")
            print(f"Диапазон дат: {results_df["Date"].min()} до {results_df["Date"].max()}")

            return results_df

        except Exception as e:
            print(f"Ошибка сохранения прогнозов: {e}")
            raise

    def run_pipeline(self, new_data_path: Path, store_data_path: Path, output_path: Path) -> pd.DataFrame:
        """Запуск полного ETL-пайплайна"""
        print("Запуск ETL-пайплайна прогнозирования спроса")

        try:
            # Extract
            raw_data = self.extract(new_data_path, store_data_path)

            # transform
            processed_data, features = self.transform(raw_data)

            # Проверка наличия данных для прогнозирования
            if len(processed_data) == 0:
                raise ValueError("Нет данных для прогнозирования после преобразований")

            # Predict
            print("Генерация прогнозов...")
            predictions = self.model.predict(processed_data[features])

            # Load
            results = self.load_predictions(predictions, output_path)

            print("ETL-пайплайн успешно завершен")

            return results

        except Exception as e:
            print(f"Критическая ошибка в ETL-пайплайне: {e}")
            raise

if __name__ == "__main__":
    print("Тестирование ETL-пайплайна...")

    pipeline = ETLPipeline()
    results = pipeline.run_pipeline(
        DATA_PATH / "raw/test.csv",
        DATA_PATH / "raw/store.csv",
        DATA_PATH / "outputs/predictions.csv"
    )

    print("Тестирование завершено успешно")