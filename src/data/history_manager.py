import logging
from datetime import timedelta, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import DATA_PATH, get_feature_config, setup_logging
from src.data.preprocessing import DataPreprocessor
from src.database.database_manager import DatabaseManager

class SalesHistoryManager:
    REQUIRED_COLUMNS = ["Store", "Date", "Sales", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]

    def __init__(self, history_file: str = "sales_history.pkl", db_path: Path | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.history_file = DATA_PATH / "processed" / history_file
        # Явный db_path для тестов, иначе дефолтный
        if db_path is not None:
            self.db = DatabaseManager(db_path=db_path)
        else:
            self.db = DatabaseManager.create_database_manager()

        self.history = self._load_history()
        self._ensure_data_integrity()

    def _ensure_data_integrity(self) -> None:
        """Проверяет загруженную историю на старте"""
        if self.history is not None and not self.history.empty:
            self._validate_data(self.history)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Store, Date, Sales и т.д. - должны быть, иначе ValueError"""
        if data is None or len(data) == 0:
            raise ValueError("Данные не могут быть None или пустыми")

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        if "Date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["Date"]):
            try:
                pd.to_datetime(data["Date"])
            except Exception as e:
                raise ValueError(f"Некорректный формат дат: {e}")

    def _load_history(self) -> pd.DataFrame:
        """БД -> pickle fallback. Начали с pickle, потом мигрировали на SQLite"""
        if self.db is not None:
            try:
                df = self.db.load_sales_history()
                if df is not None and len(df) > 0:
                    df["Date"] = pd.to_datetime(df["Date"])
                    self.logger.info(f"Загружены исторические данные из БД: {len(df)} записей")
                    return df
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки из БД, fallback на Pickle: {e}")

        return self._load_from_pickle()

    def _load_from_pickle(self) -> pd.DataFrame:
        """Pickle - legacy формат, оставлен для обратной совместимости"""
        if self.history_file.exists():
            try:
                history_data = joblib.load(self.history_file)
                if isinstance(history_data, pd.DataFrame):
                    self.logger.info(f"Загружены исторические данные из Pickle: {len(history_data)} записей")
                    return history_data
                else:
                    self.logger.warning("Файл истории содержит не DataFrame, создаем новую историю")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Ошибка загрузки исторических данных из Pickle: {e}")
                return pd.DataFrame()

        else:
            self.logger.info("Создание новой базы исторических данных...")
            return pd.DataFrame()

    def _save_history(self) -> None:
        """Дублирует в БД и pickle - на случай проблем с SQLite"""

        if self.db is not None:
            try:
                self.db.save_sales_history(self.history)
                self.logger.debug("История сохранена в БД")
            except Exception as e:
                self.logger.warning(f"Ошибка сохранения в БД: {e}")

        # Pickle backup
        try:
            if self.history is not None and not self.history.empty:
                joblib.dump(self.history, self.history_file)
                self.logger.info(f"Исторические данные сохранены: {len(self.history)} записей")
                self.logger.debug(f"Путь: {self.history_file}")
            else:
                self.logger.warning("Нет данных для сохранения")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения исторических данных в {self.history_file}: {e}")
            raise

    def update_history(self, new_data: pd.DataFrame) -> None:
        """Добавляет новые записи, дедуплицирует по Store+Date (keep last)"""
        if len(new_data) == 0:
            self.logger.warning("Попытка обновления пустыми данными")
            return

        self._validate_data(new_data)

        new_data_clean = new_data[self.REQUIRED_COLUMNS].copy()

        new_data_clean["Date"] = pd.to_datetime(new_data_clean["Date"])

        # Дедупликация: если дата уже была - берем свежие данные
        if self.history is not None and not self.history.empty:
            combined = pd.concat([self.history, new_data_clean])
            combined = combined.drop_duplicates(subset=["Store", "Date"], keep="last")
            self.history = combined.sort_values(["Store", "Date"]).reset_index(drop=True)
        else:
            self.history = new_data_clean.sort_values(["Store", "Date"]).reset_index(drop=True)

        self._save_history()
        self.logger.info(f"История обновлена: {len(new_data_clean)} новых записей, всего: {len(self.history)}")

    def get_store_history(self, store_id: int, days_back: int = 365, end_date: str | datetime = None) -> pd.DataFrame:
        """Последние N дней одного магазина до end_date"""
        if self.history is None or self.history.empty:
            return pd.DataFrame()

        store_history = self.history[self.history["Store"] == store_id].copy()

        if len(store_history) == 0:
            return pd.DataFrame()

        if end_date is None:
            end_date = store_history["Date"].max()
        else:
            end_date = pd.to_datetime(end_date)

        start_date = end_date - timedelta(days=days_back)

        filtered_history = store_history[
            (store_history["Date"] >= start_date) &
            (store_history["Date"] <= end_date)
            ].sort_values("Date")

        return filtered_history

    def calculate_lags_batch(self, store_ids: list[int], dates: list[str | datetime], lag_days: list[int] = None) -> pd.DataFrame:
        """Лаги + rolling stats из истории по полному календарю"""
        if lag_days is None:
            lag_days = get_feature_config().get("lag_days", [1, 7, 14, 28])

        store_date_pairs = pd.DataFrame({
            "Store": store_ids,
            "Date": pd.to_datetime(dates)
        })

        lag_col_names = []
        for lag in lag_days:
            lag_col_names.extend([
                f"SalesLag_{lag}",
                f"RollingMean_{lag}",
                f"RollingStd_{lag}",
                f"RollingMin_{lag}",
                f"RollingMax_{lag}"
            ])

        if self.history is None or self.history.empty:
            result_df = store_date_pairs.copy()
            for col_name in lag_col_names:
                result_df[col_name] = np.nan
            return result_df

        unique_stores = store_date_pairs["Store"].unique()
        max_lag = max(lag_days)

        df_frames = []
        dict_rows = []

        for store_id in unique_stores:
            store_mask = store_date_pairs["Store"] == store_id
            store_dates = store_date_pairs.loc[store_mask, "Date"].values

            # Берем историю с запасом max_lag*2+30 для корректных rolling
            min_date = pd.Timestamp(min(store_dates))
            history_start = min_date - timedelta(days=max_lag * 2 + 30)

            store_history = self.history[
                (self.history["Store"] == store_id) &
                (self.history["Date"] >= history_start)
            ].copy()

            if store_history.empty:
                # Новый магазин без истории
                for date in store_dates:
                    row = {"Store": store_id, "Date": pd.Timestamp(date)}
                    for col_name in lag_col_names:
                        row[col_name] = np.nan
                    dict_rows.append(row)
                continue

            store_history = store_history.sort_values("Date")
            store_df = store_history[["Date", "Sales"]].copy()
            store_df = store_df.set_index("Date")

            # Полный календарь - иначе shift сломает лаги на выходных
            max_requested = pd.Timestamp(max(store_dates))
            calendar_end = max(store_df.index.max(), max_requested)
            full_idx = pd.date_range(store_df.index.min(), calendar_end, freq="D")
            store_df = store_df.reindex(full_idx)

            # shift(1) - чтобы rolling не захватил текущий день
            shifted_sales = store_df["Sales"].shift(1)

            for lag in lag_days:
                store_df[f"SalesLag_{lag}"] = store_df["Sales"].shift(lag)

                store_df[f"RollingMean_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).mean()
                store_df[f"RollingStd_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).std()
                store_df[f"RollingMin_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).min()
                store_df[f"RollingMax_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).max()

            store_df["Store"] = store_id
            store_df = store_df.reset_index().rename(columns={"index": "Date"})
            df_frames.append(store_df)

        if len(df_frames) == 0 and len(dict_rows) == 0:
            result_df = store_date_pairs.copy()
            for col_name in lag_col_names:
                result_df[col_name] = np.nan
            return result_df

        if df_frames:
            calendar_df = pd.concat(df_frames, ignore_index=True)
        else:
            calendar_df = pd.DataFrame(columns=["Store", "Date"] + lag_col_names)

        if dict_rows:
            dict_df = pd.DataFrame(dict_rows)
            calendar_df = pd.concat([calendar_df, dict_df], ignore_index=True)

        result = store_date_pairs.merge(
            calendar_df[["Store", "Date"] + lag_col_names],
            on=["Store", "Date"],
            how="left"
        )

        return result

    def get_history_stats(self) -> dict[str, int | str | float]:
        """Кол-во записей, магазинов, покрытие по датам"""

        if self.db is not None:
            try:
                stats = self.db.get_sales_history_stats()
                if stats.get("total_records", 0) > 0:
                    return stats
            except Exception as e:
                self.logger.debug(f"Ошибка получения статистики из БД: {e}")

        if self.history is None or self.history.empty:
            return {
                "total_records": 0,
                "unique_stores": 0,
                "date_range": "База данных пуста",
                "data_size_mb": 0,
                "date_coverage_days": 0
            }

        file_size = self.history_file.stat().st_size / (1024 * 1024) if self.history_file.exists() else 0

        min_date = self.history["Date"].min()
        max_date = self.history["Date"].max()

        stats = {
            "total_records": len(self.history),
            "unique_stores": self.history["Store"].nunique(),
            "date_range": f"{min_date.strftime("%Y-%m-%d")} - {max_date.strftime("%Y-%m-%d")}",
            "data_size_mb": round(file_size, 2),
            "date_coverage_days": (max_date - min_date).days
        }

        return stats

    def get_store_stats(self, store_id: int, end_date: str | datetime | None = None) -> dict[str, int | str | float]:
        """Avg/total Sales, промо-дни, праздники за последний год"""
        store_history = self.get_store_history(store_id, days_back=365, end_date=end_date)

        if len(store_history) == 0:
            return {"error": f"Магазин {store_id} не найден в исторических данных"}

        min_date = store_history["Date"].min()
        max_date = store_history["Date"].max()

        stats = {
            "store_id": store_id,
            "records_count": len(store_history),
            "date_range": f"{min_date.strftime("%Y-%m-%d")} - {max_date.strftime("%Y-%m-%d")}",
            "avg_sales": store_history["Sales"].mean(),
            "total_sales": store_history["Sales"].sum(),
            "promo_days": store_history["Promo"].sum(),
            "holiday_days": (store_history["StateHoliday"] != "0").sum()
        }

        return stats

    def cleanup_old_data(self, cutoff_date: str | datetime) -> int:
        """Удаляет записи старше cutoff_date из БД и DataFrame"""
        if self.history is None or self.history.empty:
            self.logger.info("Нет данных для очистки")
            return 0

        initial_count = len(self.history)
        cutoff_dt = pd.to_datetime(cutoff_date)

        if self.db is not None:
            try:
                deleted = self.db.delete_old_sales_history(cutoff_dt.strftime("%Y-%m-%d"))
                self.logger.info(f"Удалено из БД: {deleted} записей")
            except Exception as e:
                self.logger.error(f"Ошибка удаления из БД: {e}")

        self.history = self.history[self.history["Date"] >= cutoff_dt].reset_index(drop=True)
        removed_count = initial_count - len(self.history)

        if removed_count > 0:
            self._save_history()
            self.logger.info(f"Удалено устаревших записей: {removed_count}")

        return removed_count

    def export_history(self, export_path: Path | None = None) -> bool:
        """Дамп истории в CSV - для бэкапа или анализа в Excel"""
        if self.history is None or self.history.empty:
            self.logger.warning("Нет данных для экспорта")
            return False

        if export_path is None:
            export_path = DATA_PATH / "processed" / "sales_history_export.csv"

        try:
            self.history.to_csv(export_path, index=False)
            self.logger.info(f"Исторические данные экспортированы: {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных: {e}")
            return False


def initialize_sales_history(data_path: Path | None = None) -> SalesHistoryManager:
    logger = logging.getLogger(__name__)

    if data_path is None:
        data_path = DATA_PATH / "raw/train.csv"

    store_path = DATA_PATH / "raw/store.csv"

    preprocessor = DataPreprocessor()
    train_data = preprocessor.load_and_merge_data(data_path, store_path)
    cleaned_data = preprocessor.clean_data(train_data)

    history_manager = SalesHistoryManager()
    history_manager.update_history(cleaned_data)

    stats = history_manager.get_history_stats()
    logger.info("База исторических данных инициализирована:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    return history_manager

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    # initialize_sales_history()

    # Тестирование функциональности
    manager = SalesHistoryManager()
    stats = manager.get_history_stats()
    logger.info("Текущая статистика истории:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
