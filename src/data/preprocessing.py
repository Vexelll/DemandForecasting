import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
from config.settings import DATA_PATH


class DataPreprocessor:
    # Константы для валидации данных
    REQUIRED_TRAIN_COLUMNS = ["Store", "Date", "Sales", "Open", "DayOfWeek", "Promo"]
    REQUIRED_STORE_COLUMNS = ["Store", "StoreType", "Assortment"]

    # Константы для валидации типов данных
    NUMERIC_COLUMNS = ["Sales", "Customers", "CompetitionDistance", "Open", "Promo"]
    INTEGER_COLUMNS = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "Promo2"]

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Настройка уровня логирования
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def load_and_merge_data(self, train_path: Path, store_path: Path) -> pd.DataFrame:
        """Загрузка и объединение данных о продажах и информации о магазинах"""
        self.logger.info(f"Загрузка данных: {train_path}, {store_path}")

        # Валидация существования файлов
        if not train_path.exists():
            raise FileNotFoundError(f"Файл с данными продаж не найден: {train_path}")
        if not store_path.exists():
            raise FileNotFoundError(f"Файл с информацией о магазинах не найден: {store_path}")

        try:
            # Загрузка данных
            train_data = pd.read_csv(train_path, low_memory=False)
            store_data = pd.read_csv(store_path, low_memory=False)

            self.logger.debug(f"Загружено данных: train={train_data.shape}, store={store_data.shape}")

        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError("Файл данных пуст") from e
        except Exception as e:
            raise Exception(f"Ошибка загрузки данных: {e}") from e

        # Валидация обязательных колонок
        self._validate_dataframe_columns(train_data, self.REQUIRED_TRAIN_COLUMNS, "train")
        self._validate_dataframe_columns(store_data, self.REQUIRED_STORE_COLUMNS, "store")

        # Объединение данных
        merged_data = pd.merge(train_data, store_data, on="Store", how="left")

        # Анализ успешности объединения
        unmatched_stores = set(train_data["Store"].unique()) - set(store_data["Store"].unique())
        if unmatched_stores:
            self.logger.warning(f"Найдены магазины без информации: {unmatched_stores}")

        self.logger.info(f"Данные объединены: {merged_data.shape[0]} записей, {merged_data.shape[1]} колонок")

        return merged_data

    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: list[str], data_type: str) -> None:
        """Валидация наличия обязательных колонок в DataFrame"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки в {data_type} данных: {missing_columns}")

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проверка и приведение типов данных"""
        df = df.copy()
        converted_columns = []

        # Числовые колонки
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if original_dtype != df[col].dtype:
                    converted_columns.append(f"{col}: {original_dtype} -> {df[col].dtype}")

        # Целочисленные колонки (только если нет NaN)
        for col in self.INTEGER_COLUMNS:
            if col in df.columns:
                # Используем Int64 для поддержки NaN в целых числах
                if df[col].notna().all():
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype("Int64")

        if converted_columns:
            self.logger.debug(f"Преобразованы типы данных: {converted_columns}")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление дубликатов по ключевым полям"""
        before_count = len(df)

        # Дубликаты по Store + Date недопустимы
        df = df.drop_duplicates(subset=["Store", "Date"], keep="last")

        removed = before_count - len(df)
        if removed > 0:
            self.logger.warning(f"Удалено дубликатов (Store + Date): {removed}")

        return df

    def _detect_outliers(self, df: pd.DataFrame, column: str = "Sales", method: str = "iqr", threshold: float = 3.0) -> pd.Series:
        """Определение выбросов в данных"""
        if column not in df.columns:
            self.logger.warning(f"Колонка {column} не найдена для детекции выбросов")
            return pd.Series([False] * len(df), index=df.index)

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

            self.logger.debug(
                f"Детекция выбросов (IQR): Q1={Q1:.2f}, Q3={Q3:.2f}, "
                f"Границы=[{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()

            if std == 0:
                self.logger.warning(f"Стандартное отклонение {column} равно 0, выбросы не определены")
                return pd.Series([False] * len(df), index=df.index)

            z_score = np.abs((df[column] - mean) / std)
            outliers = z_score > threshold

            self.logger.debug(f"Детекция выбросов (z-score): mean={mean:.2f}, std={std:.2f}, threshold={threshold}")

        else:
            raise ValueError(f"Неизвестный метод детекции выбросов: {method}. Используйте 'iqr' или 'zscore")

        outlier_count = outliers.sum()
        if outlier_count > 0:
            self.logger.info(f"Обнаружено выбросов в {column}: {outlier_count} ({outlier_count / len(df) * 100:.2f}%)")

        return outliers

    def clean_data(self, df: pd.DataFrame, remove_outliers: bool = False, outlier_method: str = "iqr", outlier_threshold: float = 3.0) -> pd.DataFrame:
        """Очистка и фильтрация данных для подготовки к анализу"""
        self.logger.info("Начало очистки данных")

        original_shape = df.shape

        # Создание копии для избежания side effects
        cleaned_df = df.copy()

        # Валидация и приведение типов данных
        cleaned_df = self._validate_data_types(cleaned_df)

        # Удаление дубликатов
        cleaned_df = self._remove_duplicates(cleaned_df)

        # Фильтрация открытых магазинов
        if "Open" in cleaned_df.columns:
            before_open = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df["Open"] == 1]
            removed_closed = before_open - len(cleaned_df)
            self.logger.debug(f"Отфильтрованы закрытые магазины: {removed_closed} записей")

        # Фильтрация валидных продаж
        before_sales = len(cleaned_df)
        sales_mask = (cleaned_df["Sales"] > 0) & (cleaned_df["Sales"].notna())
        cleaned_df = cleaned_df[sales_mask]
        removed_sales = before_sales - len(cleaned_df)
        self.logger.debug(f"Отфильтрованы невалидные продажи: {removed_sales} записей")

        # Преобразование и валидация дат
        cleaned_df = self._process_dates(cleaned_df)

        # Опциональное удаление выбросов
        if remove_outliers:
            outlier_mask = self._detect_outliers(
                cleaned_df,
                column="Sales",
                method=outlier_method,
                threshold=outlier_threshold
            )
            before_outliers = len(cleaned_df)
            cleaned_df = cleaned_df[~outlier_mask]
            removed_outliers = before_outliers - len(cleaned_df)
            self.logger.info(f"Удалено выбросов: {removed_outliers} записей")

        # Сортировка данных
        cleaned_df = cleaned_df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Анализ результатов очистки
        records_removed = original_shape[0] - len(cleaned_df)
        removal_percentage = (records_removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0

        self.logger.info(
            f"Очистка завершена: {original_shape[0]} -> {len(cleaned_df)} "
            f"(-{records_removed}, {removal_percentage:.1f}%)"
        )

        return cleaned_df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка и валидация колонки с датами"""
        if "Date" not in df.columns:
            self.logger.warning("Отсутствует колонка Date в данных")
            return df

        # Создаем копию для безопасности
        df = df.copy()

        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Подсчет и фильтрация невалидных дат
            invalid_dates = df["Date"].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Обнаружено {invalid_dates} невалидных дат, записи удалены")
                df = df[df["Date"].notna()]

            return df

        except Exception as e:
            raise ValueError(f"Ошибка преобразования дат: {e}") from e

    def get_outlier_statistics(self, df: pd.DataFrame, column: str = "Sales", method: str = "iqr", threshold: float = 3.0) -> Dict[str, Any]:
        """Получение статистики по выбросам"""
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
        """Сохранение обработанных данных в csv"""
        try:
            # Создание директории, если не существует
            path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(path, index=False)
            self.logger.info(f"Данные сохранены: {path} (записей: {len(df)})")

        except Exception as e:
            raise IOError(f"Не удалось сохранить данные в {path}: {e}") from e


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    preprocessor = DataPreprocessor(verbose=False)

    # Загрузка и объединение данных
    data = preprocessor.load_and_merge_data(DATA_PATH / "raw/train.csv", DATA_PATH / "raw/store.csv")

    # Анализ выбросов перед очисткой
    outlier_stats = preprocessor.get_outlier_statistics(data, column="Sales")
    logger.info(f"Статистика выбросов до очистки:")
    logger.info(f"Всего записей: {outlier_stats["total_records"]}")
    logger.info(f"Выбросов: {outlier_stats["outliers_count"]} ({outlier_stats["outliers_percentage"]:.2f})%")

    # Очистка данных
    cleaned_data = preprocessor.clean_data(data)

    # Сохранение результатов
    preprocessor.save_processed_data(cleaned_data, DATA_PATH / "processed/cleaned_data.csv")

    logger.info("Предобработка данных успешно завершена!")
