import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
from config.settings import DATA_PATH, setup_logging


class DataPreprocessor:
    """Загрузка train/store csv, очистка (дубли, выбросы, закрытые дни) и приведение типов"""

    REQUIRED_TRAIN_COLUMNS = ["Store", "Date", "Sales", "Open", "DayOfWeek", "Promo"]
    REQUIRED_STORE_COLUMNS = ["Store", "StoreType", "Assortment"]
    NUMERIC_COLUMNS = ["Sales", "Customers", "CompetitionDistance", "Open", "Promo"]
    INTEGER_COLUMNS = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "Promo2"]

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Варианты логирования: DEBUG или INFO(дефолт)
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def load_and_merge_data(self, train_path: Path, store_path: Path) -> pd.DataFrame:
        """train.csv + store.csv -> один DataFrame по ключу Store"""
        self.logger.info(f"Загрузка: {train_path}, {store_path}")

        if not train_path.exists():
            raise FileNotFoundError(f"Файл продаж не найден: {train_path}")
        if not store_path.exists():
            raise FileNotFoundError(f"Файл магазинов не найден: {store_path}")

        try:
            train_data = pd.read_csv(train_path, low_memory=False)
            store_data = pd.read_csv(store_path, low_memory=False)
            self.logger.debug(f"Размеры: train={train_data.shape}, store={store_data.shape}")
        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError("Файл данных пуст") from e
        except Exception as e:
            raise Exception(f"Ошибка загрузки: {e}") from e

        self._validate_dataframe_columns(train_data, self.REQUIRED_TRAIN_COLUMNS, "train")
        self._validate_dataframe_columns(store_data, self.REQUIRED_STORE_COLUMNS, "store")

        merged_data = pd.merge(train_data, store_data, on="Store", how="left")

        # Если магазин есть в train, но нет в store - признаки будут NaN
        unmatched_stores = set(train_data["Store"].unique()) - set(store_data["Store"].unique())
        if unmatched_stores:
            self.logger.warning(f"Есть магазины без информации: {unmatched_stores}")

        self.logger.info(f"Merge готов: {merged_data.shape[0]} записей, {merged_data.shape[1]} колонок")

        return merged_data

    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: list[str], data_type: str) -> None:
        """Проверяем, что нужные колонки есть, иначе ValueError"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Нет обязательных колонок в {data_type}: {missing_columns}")

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приведение типов: числовые -> numeric, целые -> int/Int64"""
        df = df.copy()
        converted_columns = []

        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if original_dtype != df[col].dtype:
                    converted_columns.append(f"{col}: {original_dtype} -> {df[col].dtype}")

        for col in self.INTEGER_COLUMNS:
            if col in df.columns:
                # Int64 вместо int - поддерживает NaN без падения
                if df[col].notna().all():
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype("Int64")

        if converted_columns:
            self.logger.debug(f"Типы изменены: {converted_columns}")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Store + Date должны быть уникальны, иначе лаги поедут"""
        before_count = len(df)
        df = df.drop_duplicates(subset=["Store", "Date"], keep="last")

        removed = before_count - len(df)
        if removed > 0:
            self.logger.warning(f"Дубликаты (Store+Date): удалено {removed}")

        return df

    def _detect_outliers(self, df: pd.DataFrame, column: str = "Sales", method: str = "iqr", threshold: float = 3.0) -> pd.Series:
        """IQR или z-score детекция, возвращает bool маску"""
        if column not in df.columns:
            self.logger.warning(f"Колонка {column} не найдена")
            return pd.Series([False] * len(df), index=df.index)

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

            self.logger.debug(
                f"IQR: Q1={Q1:.2f}, Q3={Q3:.2f}, "
                f"Границы=[{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()

            if std == 0:
                self.logger.warning(f"std({column}) = 0, выбросы не определить")
                return pd.Series([False] * len(df), index=df.index)

            z_score = np.abs((df[column] - mean) / std)
            outliers = z_score > threshold
            self.logger.debug(f"z-score: mean={mean:.2f}, std={std:.2f}, порог={threshold}")

        else:
            raise ValueError(f"Неизвестный метод: {method}. Используйте 'iqr' или 'zscore")

        outlier_count = outliers.sum()
        if outlier_count > 0:
            self.logger.info(f"Выбросов в {column}: {outlier_count} ({outlier_count / len(df) * 100:.2f}%)")

        return outliers

    def clean_data(self, df: pd.DataFrame, remove_outliers: bool = False, outlier_method: str = "iqr", outlier_threshold: float = 3.0) -> pd.DataFrame:
        """Основная очистка: типы -> дубли -> закрытые -> нулевые Sales -> даты -> выбросы"""
        self.logger.info("Очистка данных...")
        original_count = df.shape[0]

        cleaned_df = df.copy()
        cleaned_df = self._validate_data_types(cleaned_df)
        cleaned_df = self._remove_duplicates(cleaned_df)

        # Закрытые магазины: Sales всегда 0, для обучения бесполезны
        if "Open" in cleaned_df.columns:
            before_open = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df["Open"] == 1]
            self.logger.debug(f"Закрытые магазины: -{before_open - len(cleaned_df)}")

        # Нулевые/NaN продажи - нет смысла учить модель на них
        before_sales = len(cleaned_df)
        cleaned_df = cleaned_df[(cleaned_df["Sales"] > 0) & (cleaned_df["Sales"].notna())]
        self.logger.debug(f"Невалидные Sales: -{before_sales - len(cleaned_df)} записей")

        cleaned_df = self._process_dates(cleaned_df)

        if remove_outliers:
            outlier_mask = self._detect_outliers(
                cleaned_df, column="Sales",
                method=outlier_method, threshold=outlier_threshold
            )
            before_outliers = len(cleaned_df)
            cleaned_df = cleaned_df[~outlier_mask]
            self.logger.info(f"Выбросы удалены: -{before_outliers - len(cleaned_df)} записей")

        # Сортировка по магазину и дате - важно для корректных лагов дальше
        cleaned_df = cleaned_df.sort_values(["Store", "Date"]).reset_index(drop=True)

        removed = original_count - len(cleaned_df)
        pct = (removed / original_count * 100) if original_count > 0 else 0
        self.logger.info(f"Очистка: {original_count} -> {len(cleaned_df)} (-{removed}, {pct:.1f}%)")

        return cleaned_df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Date -> datetime, невалидные даты выкидываются"""
        if "Date" not in df.columns:
            self.logger.warning("Нет колонки Date")
            return df

        df = df.copy()

        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            invalid_dates = df["Date"].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Невалидных дат: {invalid_dates}, удалены")
                df = df[df["Date"].notna()]

            return df

        except Exception as e:
            raise ValueError(f"Ошибка парсинга дат: {e}") from e

    def get_outlier_statistics(self, df: pd.DataFrame, column: str = "Sales", method: str = "iqr", threshold: float = 3.0) -> Dict[str, Any]:
        """Статистика выбросов: сколько, какие границы, распределение"""
        outlier_mask = self._detect_outliers(df, column, method, threshold)
        outliers = df[outlier_mask][column]
        normal = df[~outlier_mask][column]

        stats = {
            "total_records": len(df),
            "outliers_count": outlier_mask.sum(),
            "outliers_percentage": (outlier_mask.sum() / len(df)) * 100 if len(df) > 0 else 0,
            "method": method,
            "threshold": threshold,
            "normal_stats": {
                "min": normal.min() if len(normal) > 0 else None,
                "max": normal.max() if len(normal) > 0 else None,
                "mean": normal.mean() if len(normal) > 0 else None,
                "median": normal.median() if len(normal) > 0 else None
            },
            "outlier_stats": {
                "min": outliers.min() if len(outliers) > 0 else None,
                "max": outliers.max() if len(outliers) > 0 else None,
                "mean": outliers.mean() if len(outliers) > 0 else None
            }
        }

        return stats

    def save_processed_data(self, df: pd.DataFrame, path: Path) -> None:
        """Дамп в csv, создает директорию, если нет"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            self.logger.info(f"Сохранено: {path} ({len(df)} записей)")
        except Exception as e:
            raise IOError(f"Не удалось сохранить данные в {path}: {e}") from e


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    preprocessor = DataPreprocessor(verbose=False)

    data = preprocessor.load_and_merge_data(DATA_PATH / "raw/train.csv", DATA_PATH / "raw/store.csv")

    # Смотрим на выбросы до очистки
    outlier_stats = preprocessor.get_outlier_statistics(data, column="Sales")
    logger.info(f"Статистика выбросов до очистки:")
    logger.info(f"Записей: {outlier_stats["total_records"]}")
    logger.info(f"Выбросов: {outlier_stats["outliers_count"]} ({outlier_stats["outliers_percentage"]:.2f}%)")

    cleaned_data = preprocessor.clean_data(data)
    preprocessor.save_processed_data(cleaned_data, DATA_PATH / "processed/cleaned_data.csv")

    logger.info("Предобработка завершена")
