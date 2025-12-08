import pandas as pd
from pathlib import Path
from config.settings import DATA_PATH


class DataPreprocessor:
    # Константы для валидации данных
    REQUIRED_TRAIN_COLUMNS = ["Store", "Date", "Sales", "Open", "DayOfWeek", "Promo"]
    REQUIRED_STORE_COLUMNS = ["Store", "StoreType", "Assortment"]

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def load_and_merge_data(self, train_path: Path, store_path: Path) -> pd.DataFrame:
        """Загрузка и объединение данных о продажах и информации о магазинах"""
        if self.verbose:
            print(f"Загрузка данных: {train_path}, {store_path}")

        # Валидация существования файлов
        if not train_path.exists():
            raise FileNotFoundError(f"Файл с данными продаж не найден: {train_path}")
        if not store_path.exists():
            raise FileNotFoundError(f"Файл с информацией о магазинах не найден: {store_path}")

        try:
            # Загрузка данных
            train_data = pd.read_csv(train_path, low_memory=False)
            store_data = pd.read_csv(store_path, low_memory=False)

            if self.verbose:
                print(f"Загружено данных: train={train_data.shape}, store={store_data.shape}")

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
        if unmatched_stores and self.verbose:
            print(f"Предупреждение: найдены магазины без информации: {unmatched_stores}")

        if self.verbose:
            print(f"Данные объединены: {merged_data}")

        return merged_data

    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: list[str], data_type: str) -> None:
        """Валидация наличия обязательных колонок в DataFrame"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки в {data_type} данных: {missing_columns}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка и фильтрация данных для подготовки к анализу"""
        if self.verbose:
            print("Начало очистки данных")

        original_shape = df.shape

        # Создание копии для избежания side effects
        cleaned_df = df.copy()

        # Фильтрация открытых магазинов
        if "Open" in cleaned_df.columns:
            before_open = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df["Open"] == 1]
            if self.verbose:
                print(f"Отфильтрованы закрытые магазины: {before_open} -> {len(cleaned_df)}")

        # Фильтрация валидных продаж
        sales_mask = (cleaned_df["Sales"] > 0) & (cleaned_df["Sales"].notna())
        before_sales = len(cleaned_df)
        cleaned_df = cleaned_df[sales_mask]
        if self.verbose:
            print(f"Отфильтрованы невалидные продажи: {before_sales} -> {len(cleaned_df)}")

        # Преобразование и валидация дат
        cleaned_df = self._process_dates(cleaned_df)

        # Сортировка данных
        cleaned_df = cleaned_df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Анализ результатов очистки
        records_removed = original_shape[0] - len(cleaned_df)
        if self.verbose:
            removal_percentage = (records_removed / original_shape[0]) * 100
            print(f"Очистка завершена: {original_shape[0]} -> {len(cleaned_df)} (-{records_removed}, {removal_percentage:.1f}%)")

        return cleaned_df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка и валидация колонки с датами"""
        if "Date" not in df.columns:
            if self.verbose:
                print("Предупреждение: отсутствует колонка Date в данных")
            return df

        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Проверка на неудачное преобразование
            invalid_dates = df["Date"].isna().sum()
            if invalid_dates > 0 and self.verbose:
                print(f"Предупреждение: обнаружено {invalid_dates} невалидных дат")
                df = df[df["Date"].notna()]

            return df

        except Exception as e:
            raise Exception(f"Ошибка преобразования дат: {e}") from e

    def save_processed_data(self, df: pd.DataFrame, path: Path) -> None:
        """Сохранение обработанных данных в csv"""
        try:
            # Создание директории если не существует
            path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(path, index=False)
            if self.verbose:
                print(f"Данные сохранены: {path} (записей: {len(df)})")

        except Exception as e:
            raise IOError(f"Не удалось сохранить данные в {path}: {e}") from e


if __name__ == "__main__":
    preprocessor = DataPreprocessor(verbose=True)

    data = preprocessor.load_and_merge_data(DATA_PATH / "raw/train.csv", DATA_PATH / "raw/store.csv")
    cleaned_data = preprocessor.clean_data(data)
    preprocessor.save_processed_data(cleaned_data, DATA_PATH / "processed/cleaned_data.csv")

    print("Предобработка данных успешно завершена!")