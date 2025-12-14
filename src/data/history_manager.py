import pandas as pd
import numpy as np
import joblib
from typing import Optional, Union, List
from datetime import timedelta, datetime
from pathlib import Path
from config.settings import DATA_PATH
from src.data.preprocessing import DataPreprocessor

class SalesHistoryManager:
    REQUIRED_COLUMNS = ["Store", "Date", "Sales", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]

    def __init__(self, history_file: str = "sales_history.pkl") -> None:
        self.history_file = DATA_PATH / "processed" / history_file
        self.history = self._load_history()
        self._ensure_data_integrity()

    def _ensure_data_integrity(self) -> None:
        """Гарантирует целостность исторических данных"""
        if self.history is not None and not self.history.empty:
            self._validate_data(self.history)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Валидация данных на наличие обязательных колонок и корректность типов"""
        if data is None or len(data) == 0:
            raise ValueError("Данные не могут быть None или пустыми")

        # Проверка обязательных колонок
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        # Проверка формата дат
        if "Date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["Date"]):
            try:
                pd.to_datetime(data["Date"])
            except Exception as e:
                raise ValueError(f"Некорректный формат дат: {e}")

    def _load_history(self) -> pd.DataFrame:
        """Загрузка истории из файла"""
        if self.history_file.exists():
            try:
                history_data = joblib.load(self.history_file)
                if isinstance(history_data, pd.DataFrame):
                    print(f"Загружены исторические данные: {len(history_data)} записей")
                    return history_data
                else:
                    print("Предупреждение: файл истории содержит не DataFrame, создаем новую историю")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Ошибка загрузки исторических данных: {e}")
                return pd.DataFrame()

        else:
            print("Создание новой базы исторических данных...")
            return pd.DataFrame()

    def _save_history(self) -> None:
        """Сохранение истории в файл"""
        try:
            if self.history is not None and not self.history.empty:
                joblib.dump(self.history, self.history_file)
                print(f"Исторические данные сохранены: {len(self.history)} записей")
                print(f"Путь: {self.history_file}")
            else:
                print("Предупреждение: нет данных для сохранения")
        except Exception as e:
            print(f"Ошибка сохранения исторических данных в {self.history_file}: {e}")
            raise

    def update_history(self, new_data: pd.DataFrame) -> None:
        """Обновление истории новыми данными"""
        if len(new_data) == 0:
            print("Предупреждение: попытка обновления пустыми данными")
            return

        # Валидация входных данных
        self._validate_data(new_data)

        # Выбираем только нужные колонки
        new_data_clean = new_data[self.REQUIRED_COLUMNS].copy()

        # Преобразование дат
        new_data_clean["Date"] = pd.to_datetime(new_data_clean["Date"])

        # Объединение с существующей историей
        if self.history is not None and not self.history.empty:
            # Удаляем дубликаты по Store и Date, сохраняя последние данные
            combined = pd.concat([self.history, new_data_clean])
            combined = combined.drop_duplicates(subset=["Store", "Date"], keep="last")
            self.history = combined.sort_values(["Store", "Date"]).reset_index(drop=True)
        else:
            self.history = new_data_clean.sort_values(["Store", "Date"]).reset_index(drop=True)

        self._save_history()
        print(f"История обновлена: {len(new_data_clean)} новых записей, всего: {len(self.history)}")

    def get_store_history(self, store_id: int, days_back: int = 365, end_date: Union[str, datetime, None] = None) -> pd.DataFrame:
        """Получение истории продаж для конкретного магазина"""
        if self.history is None or self.history.empty:
            return pd.DataFrame()

        store_history = self.history[self.history["Store"] == store_id].copy()

        if len(store_history) == 0:
            return pd.DataFrame()

        # Определение конечной даты
        if end_date is None:
            end_date = store_history["Date"].max()
        else:
            end_date = pd.to_datetime(end_date)

        # Фильтрация по временному диапазону
        start_date = end_date - timedelta(days=days_back)

        filtered_history = store_history[
            (store_history["Date"] >= start_date) &
            (store_history["Date"] <= end_date)
            ].sort_values("Date")

        return filtered_history

    def get_latest_sales(self, store_id: int, days: int = 28, end_date: Union[str, datetime, None] = None) -> Optional[pd.Series]:
        """Получение последних продаж для магазина до указанной даты"""
        store_history = self.get_store_history(store_id, days_back=days + 7, end_date=end_date)

        if len(store_history) == 0:
            return None

        # Определение конечной даты
        if end_date is None:
            end_date = store_history["Date"].max()
        else:
            end_date = pd.to_datetime(end_date)

        # Получение последних days дней ДО end_date
        start_date = end_date - timedelta(days=days - 1)

        recent_sales = store_history[
            (store_history["Date"] >= start_date) &
            (store_history["Date"] < end_date)
            ][["Date", "Sales"]].set_index("Date")["Sales"]

        return recent_sales

    def calculate_lags_batch(self, store_ids: list[int], dates: List[Union[str, datetime]], lag_days: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
        """Расчет лаговых признаков"""
        results = []

        # Группируем по магазинам для оптимизации
        store_date_pairs = pd.DataFrame({
            "Store": store_ids,
            "Date": pd.to_datetime(dates)
        })

        # Для каждого уникального магазина
        unique_stores = store_date_pairs["Store"].unique()

        for store_id in unique_stores:
            # Фильтруем данные для текущего магазина
            store_mask = store_date_pairs["Store"] == store_id
            store_dates = store_date_pairs.loc[store_mask, "Date"]

            max_lag = max(lag_days)

            # Для каждой даты в запросе
            for date in store_dates:
                result = {"Store": store_id, "Date": date}

                history_end = date - timedelta(days=1) # История до целевой даты

                history = self.get_store_history(
                    store_id,
                    days_back=max_lag * 2 + 30,  # Запас для расчета
                    end_date=history_end
                )

                if history.empty:
                    # Если истории нет, заполняем NaN только для этой даты
                    for lag in lag_days:
                        result[f"SalesLag_{lag}"] = np.nan
                        result[f"RollingMean_{lag}"] = np.nan
                        result[f"RollingStd_{lag}"] = np.nan
                    results.append(result)
                    continue

                # Сортируем историю по дате
                history = history.sort_values("Date")
                history_ts = history.set_index("Date")["Sales"]

                for lag in lag_days:
                    # Лаговые признаки
                    lag_date = date - timedelta(days=lag)
                    lag_value = history_ts.get(lag_date, np.nan)
                    result[f"SalesLag_{lag}"] = lag_value

                    # Скользящее среднее и STD
                    window_end = date - timedelta(days=1)
                    window_start = window_end - timedelta(days=lag - 1)

                    window_data = history_ts[
                        (history_ts.index >= window_start) &
                        (history_ts.index <= window_end)
                    ]

                    result[f"RollingMean_{lag}"] = window_data.mean() if len(window_data) > 0 else np.nan

                    result[f"RollingStd_{lag}"] = window_data.std() if len(window_data) > 1 else np.nan



                results.append(result)

        return pd.DataFrame(results)

    def get_history_stats(self) -> dict[str, int | str | float]:
        """Статистика по конкретному магазину на определенную дату"""
        if self.history is None or self.history.empty:
            return {
                "total_records": 0,
                "unique_stores": 0,
                "date_range": "База данных пуста",
                "data_size_mb": 0,
                "date_coverage_days": 0
            }

        file_size = self.history_file.stat().st_size / (1024 * 1024) if self.history_file.exists() else 0

        stats = {
            "total_records": len(self.history),
            "unique_stores": self.history["Store"].nunique(),
            "date_range": f"{self.history["Date"].min().strftime("%Y-%m-%d")} - {self.history["Date"].max().strftime("%Y-%m-%d")}",
            "data_size_mb": round(file_size, 2),
            "date_coverage_days": (self.history["Date"].max() - self.history["Date"].min()).days
        }

        return stats

    def get_store_stats(self, store_id: int, end_date: Optional[str | datetime] = None) -> dict[str, int | str | float]:
        """Статистика по конкретному магазину на определенную дату"""
        store_history = self.get_store_history(store_id, days_back=365, end_date=end_date)

        if len(store_history) == 0:
            return {"error": f"Магазин {store_id} не найден в исторических данных"}

        stats = {
            "store_id": store_id,
            "records_count": len(store_history),
            "date_range": f"{store_history["Date"].min().strftime("%Y-%m-%d")} - {store_history["Date"].max().strftime("%Y-%m-%d")}",
            "avg_sales": store_history["Sales"].mean(),
            "total_sales": store_history["Sales"].sum(),
            "promo_days": store_history["Promo"].sum(),
            "holiday_days": (store_history["StateHoliday"] != "0").sum()
        }

        return stats

    def cleanup_old_data(self, cutoff_date: str | datetime) -> int:
        """Очистка устаревших данных"""
        if self.history is None or self.history.empty:
            print("Нет данных для очистки")
            return 0

        initial_count = len(self.history)
        cutoff_dt = pd.to_datetime(cutoff_date)

        self.history = self.history[self.history["Date"] >= cutoff_dt]
        removed_count = initial_count - len(self.history)

        if removed_count > 0:
            self._save_history()
            print(f"Удалено устаревших записей: {removed_count}")

        return removed_count

    def export_history(self, export_path: Optional[Path] = None) -> bool:
        """Экспорт исторических данных в CSV"""
        if self.history is None or self.history.empty:
            print("Нет данных для экспорта")
            return False

        if export_path is None:
            export_path = DATA_PATH / "processed" / "sales_history_export.csv"

        try:
            self.history.to_csv(export_path, index=False)
            print(f"Исторические данные экспортированы: {export_path}")
            return True
        except Exception as e:
            print(f"Ошибка экспорта данных: {e}")
            return False


def initialize_sales_history(data_path: Optional[Path] = None) -> SalesHistoryManager:
        if data_path is None:
            data_path = DATA_PATH / "raw/train.csv"

        store_path = DATA_PATH / "raw/store.csv"

        preprocessor = DataPreprocessor()
        train_data = preprocessor.load_and_merge_data(data_path, store_path)
        cleaned_data = preprocessor.clean_data(train_data)

        history_manager = SalesHistoryManager()
        history_manager.update_history(cleaned_data)

        stats = history_manager.get_history_stats()
        print("База исторических данных инициализирована:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        return history_manager

if __name__ == "__main__":
    initialize_sales_history()

    # Тестирование функциональности
    manager = SalesHistoryManager()
    stats = manager.get_history_stats()
    print("Текущая статистика истории:")
    for key, value in stats.items():
        print(f"{key}: {value}")