import pytest
import numpy as np
import pandas as pd
import logging
import tempfile
from pathlib import Path
import shutil
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.data.history_manager import SalesHistoryManager

logger = logging.getLogger(__name__)

class TestDataPreprocessor:
    """Тесты для модуля предобработки данных"""

    def setup_method(self):
        """Инициализация тестового окружения"""
        self.preprocessor = DataPreprocessor()
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Очистка тестового окружения"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_and_merge_data_basic(self):
        """Базовый тест загрузки и объединения данных"""
        test_train = pd.DataFrame({
            "Store": [1, 1, 2, 2],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "Sales": [1000, 1200, 800, 900],
            "Open": [1, 1, 1, 1],
            "DayOfWeek": [1, 2, 1, 2],
            "Promo": [1, 0, 1, 0],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 0, 0]
        })

        test_store = pd.DataFrame({
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "b"],
            "CompetitionDistance": [1000.0, 1500.0],
            "CompetitionOpenSinceMonth": [1.0, 2.0],
            "CompetitionOpenSinceYear": [2023.0, 2023.0],
            "Promo2": [1, 0],
            "Promo2SinceWeek": [1.0, 0.0],
            "Promo2SinceYear": [2023.0, 0.0],
            "PromoInterval": ["Jan,Apr,Jul,Oct", "NoPromo"]
        })

        # Сохранение временных файлов
        test_train_path = self.test_dir / "train.csv"
        test_store_path = self.test_dir / "store.csv"

        test_train.to_csv(test_train_path, index=False)
        test_store.to_csv(test_store_path, index=False)

        # Тестируем метод
        result = self.preprocessor.load_and_merge_data(test_train_path, test_store_path)

        # Проверяем результаты
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert "StoreType" in result.columns
        assert "Assortment" in result.columns
        assert "CompetitionDistance" in result.columns
        assert result["Sales"].sum() == 3900

    def test_load_and_merge_data_missing_files(self):
        """Тест обработки отсутствующих файлов"""
        missing_train_path = self.test_dir / "missing_train.csv"
        missing_store_path = self.test_dir / "missing_store.csv"

        with pytest.raises(FileNotFoundError):
            self.preprocessor.load_and_merge_data(missing_train_path, missing_store_path)

    def test_load_and_merge_data_missing_columns(self):
        """Тест обработки данных с отсутствующими обязательными колонками"""
        test_train = pd.DataFrame({
            "Store": [1, 2],
            "Date": ["2024-01-01", "2024-01-02"],
            # Отсутствует колонка Sales
            "Open": [1, 1],
            "DayOfWeek": [1, 2]
        })

        test_store = pd.DataFrame({
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "b"]
        })

        test_train_path = self.test_dir / "train_invalid.csv"
        test_store_path = self.test_dir / "store_invalid.csv"

        test_train.to_csv(test_train_path, index=False)
        test_store.to_csv(test_store_path, index=False)

        with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
            self.preprocessor.load_and_merge_data(test_train_path, test_store_path)

    def test_clean_data_basic(self):
        """Базовый тест очистки данных"""
        test_data = pd.DataFrame({
            "Store": [1, 1, 2, 2],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "Sales": [1000, 0, 800, 900], # Одна нулевая продажа
            "Open": [1, 1, 0, 1], # Один закрытый магазин
            "DayOfWeek": [1, 2, 1, 2],
            "Promo": [1, 0, 1, 0],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 0, 0]
        })

        result = self.preprocessor.clean_data(test_data)

        # Должны остаться только открытые магазины с продажами > 0
        assert len(result) == 2
        assert (result["Sales"] > 0).all()
        assert (result["Open"] == 1).all()
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_clean_data_with_invalid_dates(self):
        """Тест очистки данных с невалидными датами"""
        test_data = pd.DataFrame({
            "Store": [1, 1],
            "Date": ["invalid-date", "2024-01-02"], # Одна невалидная дата
            "Sales": [1000, 1200],
            "Open": [1, 1],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        result = self.preprocessor.clean_data(test_data)

        # Должна остаться только одна запись с валидной датой
        assert len(result) == 1
        assert result.iloc[0]["Date"] == pd.Timestamp("2024-01-02")


    def test_save_processed_data(self):
        """Тест сохранения обработанных данных"""
        test_data = pd.DataFrame({
            "Store": [1, 2],
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [1000, 1200],
            "Open": [1, 1],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        save_path = self.test_dir / "proccessed/test_output.csv"

        # Преобразуем даты для clean_data
        test_data["Date"] = pd.to_datetime(test_data["Date"])
        cleaned_data = self.preprocessor.clean_data(test_data)

        self.preprocessor.save_processed_data(cleaned_data, save_path)

        assert save_path.exists()

        # Проверяем, что файл можно загрузить обратно
        loaded_data = pd.read_csv(save_path, parse_dates=["Date"])
        assert len(loaded_data) == 2
        assert "Sales" in loaded_data.columns

class TestFeatureEngineer:
    """Тесты для feature engineering"""

    def setup_method(self):
        """Инциализация тестового окружения"""
        self.engineer = FeatureEngineer()

    def test_temporal_features_creation(self):
        """Тест создания временных признаков"""
        test_data = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-25"]),
            "DayOfWeek": [1, 6, 3], # Понедельник, суббота, среда
            "Sales": [1000, 1200, 800],
            "Store": [1, 1, 1]
        })

        result = self.engineer.create_temporal_features(test_data)

        # Проверяем созданные признаки
        expected_columns = ["Year", "Month", "Week", "DayOfYear",
                          "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
                          "IsSaturday", "IsSunday", "IsSpring", "IsSummer",
                          "IsAutumn", "IsWinter"]

        for col in expected_columns:
            assert col in result.columns, f"Отсутствует колонка: {col}"

        assert result["Year"].iloc[0] == 2024
        assert result["Month"].iloc[0] == 1
        assert result["Month"].iloc[1] == 6
        assert result["IsSaturday"].iloc[0] == 0
        assert result["IsSaturday"].iloc[1] == 1 # Первая дата - понедельник
        assert result["IsWinter"].iloc[0] == 1 # Вторая дата - суббота
        assert result["IsSummer"].iloc[1] == 1 # Июнь - лето

    def test_handle_nan_values_comprehensive(self):
        """Тест обработки пропущенных значений"""
        test_date = pd.DataFrame({
            "CompetitionDistance": [1000.0, np.nan, 1500.0, np.nan],
            "CompetitionOpenSinceMonth": [1.0, np.nan, 2.0, np.nan],
            "CompetitionOpenSinceYear": [2015.0, np.nan, 2016.0, np.nan],
            "Promo2SinceWeek": [1.0, np.nan, 2.0, np.nan],
            "Promo2SinceYear": [2020.0, np.nan, 2021.0, np.nan],
            "PromoInterval": ["Jan,Apr", np.nan, "Feb,May", np.nan],
            "Sales": [1000.0, 1200.0, 800.0, 900.0],
            "Store": [1, 2, 3, 4],
            "StoreType": ["a", "b", "a", "b"],
            "Assortment": ["a", "b", "a", "b"]
        })

        result = self.engineer.handle_nan_values(test_date)

        # Проверяем, что нет NaN значений
        assert result.isna().sum().sum() == 0, "В данных остались NaN значения"

        # Проверяем создание бинарных признаков
        assert "HasCompetition" in result.columns
        assert "InPromo2" in result.columns

        # Проверяем логику заполнения
        assert result["HasCompetition"].iloc[1] == 0 # NaN -> нет конкурентов
        assert result["HasCompetition"].iloc[0] == 1 # Есть значение -> есть конкуренты

        assert result["InPromo2"].iloc[1] == 0 # NaN -> не участвует в Promo2
        assert result["InPromo2"].iloc[0] == 1 # Есть значение -> участвует в Promo2

        # Проверяем что PromoInterval заполнен
        assert result["PromoInterval"].iloc[1] == "NoPromo"

    def test_create_promo_features(self):
        """Тест создания признаков связанных с акциями"""
        test_data = pd.DataFrame({
            "Store": [1, 1, 1, 1, 2, 2, 2, 2],
            "Date": pd.date_range("2024-01-01", periods=8),
            "Sales": [1000, 1200, 800, 900, 1100, 1300, 850, 950],
            "Promo": [1, 1, 0, 0, 1, 0, 1, 0] # Последовательности промо
        })

        # Сортируем по Store и Date
        test_data = test_data.sort_values(["Store", "Date"]).reset_index(drop=True)

        result = self.engineer.create_promo_features(test_data)

        # Проверяем созданные признаки
        assert "PromoSequence" in result.columns
        assert "PromoSequenceLength" in result.columns
        assert "DaysSinceLastPromo" in result.columns
        assert "IsPromoStart" in result.columns
        assert "IsPromoEnd" in result.columns

        # Проверяем логику
        store_1_data = result[result["Store"] == 1]

        # Проверяем длину последовательности промо
        assert "PromoSequenceLength" in store_1_data.columns
        assert "IsPromoStart" in store_1_data.columns

        # Проверяем, что IsPromoStart определен для первой записи магазина
        assert store_1_data["IsPromoStart"].iloc[0] in [0, 1]

    def test_create_holiday_features(self):
        """Тест создания праздничных признаков"""
        test_data = pd.DataFrame({
            "StateHoliday": ["0", "a", "b", "c", "0", "a"], # Разные типы праздников
            "Store": [1, 1, 2, 2, 3, 3],
            "Date": pd.date_range("2024-01-01", periods=6),
            "Sales": [1000, 1200, 800, 900, 1100, 1300]
        })

        result = self.engineer.create_holiday_features(test_data)

        # Проверяем созданные признаки
        assert "IsHoliday" in result.columns
        assert "IsPublicHoliday" in result.columns
        assert "IsEaster" in result.columns
        assert "IsChristmas" in result.columns

        # Проверяем логику
        assert result["IsHoliday"].sum() == 4 # Все кроме "0"
        assert result["IsPublicHoliday"].sum() == 2 # только "a"
        assert result["IsEaster"].sum() == 1 # Только "b"
        assert result["IsChristmas"].sum() == 1 # Только "c"

    def test_prepare_final_dataset_integration(self):
        """Интегрированный тест подготовки финального датасета"""
        # Создаем реалистичные тестовые данные
        base_date = pd.Timestamp("2024-01-01")
        test_data = pd.DataFrame({
            "Store": [1, 1, 2, 2],
            "Date": [base_date, base_date + pd.Timedelta(days=1),
                     base_date, base_date + pd.Timedelta(days=1)],
            "Sales": [1000.0, 1200.0, 800.0, 900.0],
            "DayOfWeek": [1, 2, 1, 2],
            "Promo": [1, 0, 1, 0],
            "StateHoliday": ["0", "0", "a", "0"],
            "SchoolHoliday": [0, 0, 1, 0],
            "StoreType": ["a", "a", "b", "b"],
            "Assortment": ["a", "a", "b", "b"],
            "CompetitionDistance": [1000.0, 1000.0, 1500.0, 1500.0],
            "CompetitionOpenSinceMonth": [1.0, 1.0, 2.0, 2.0],
            "CompetitionOpenSinceYear": [2023.0, 2023.0, 2023.0, 2023.0],
            "Promo2": [1, 1, 0, 0],
            "Promo2SinceWeek": [1.0, 1.0, 0.0, 0.0],
            "Promo2SinceYear": [2023.0, 2023.0, 0.0, 0.0],
            "PromoInterval": ["Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct", "NoPromo", "NoPromo"]
        })

        # Запуск полного пайплайна
        result, features = self.engineer.prepare_final_dataset(test_data)

        # Проверяем результаты
        assert isinstance(result, pd.DataFrame)
        assert isinstance(features, list)
        assert len(features) > 0

        # Проверяем, что целевая переменная присутствует
        assert "Sales" in result.columns

        # Проверяем, что признаки созданы
        assert len(result.columns) > 10

        # Проверяем отсутствие пропусков в признаках
        if len(features) > 0:
            # Лаговые признаки могут иметь NaN для первых записей
            non_lag_features = [f for f in features if "Lag" not in f and "Rolling" not in f]
            if non_lag_features:
                assert result[features].isna().sum().sum() == 0, f"Найдены пропуски в признаках: {result[features].isna().sum().to_dict()}"

        # Проверяем статистику
        stats = self.engineer.get_feature_statistics()
        assert stats["total_features"] == len(features)
        assert stats["processed_records"] == len(result)

class TestHistoryManager:
    """Тесты для менеджера истории"""

    def setup_method(self):
        """Инициализация тестового окружения"""
        self.test_dir = Path(tempfile.mkdtemp())
        # Путь для pickle-файла
        self.history_file = self.test_dir / "test_history.pkl"
        # Путь для временной БД
        self.db_path = self.test_dir / "test_history.db"
        self.manager = SalesHistoryManager(
            history_file = str(self.history_file),
            db_path = self.db_path
        )

    def teardown_method(self):
        """Очистка тестового окружения"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_history_operations_basic(self):
        """Базовые тесты операций с историей с проверкой сохранения в БД"""
        test_data = pd.DataFrame({
            "Store": [1, 1, 2],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "Sales": [1000.0, 1200.0, 800.0],
            "DayOfWeek": [1, 2, 1],
            "Promo": [1, 0, 1],
            "StateHoliday": ["0", "0", "0"],
            "SchoolHoliday": [0, 0, 0]
        })

        # Тест обновления истории
        self.manager.update_history(test_data)

        # Проверяем, что история сохранена (файл .pkl может отсутствовать)
        # Но данные должны быть в БД
        assert self.db_path.exists()
        assert len(self.manager.history) == 3

        # Тест получения истории магазина
        store_history = self.manager.get_store_history(1)
        assert len(store_history) == 2
        assert (store_history["Store"] == 1).all()
        assert store_history["Sales"].sum() == 2200.0

        # Тест статистики
        stats = self.manager.get_history_stats()
        assert stats["total_records"] == 3
        assert stats["unique_stores"] == 2
        assert "date_range" in stats

        # Тест расчета лагов
        lags = self.manager.calculate_lags_batch([1], ["2024-01-03"])
        assert isinstance(lags, pd.DataFrame)
        assert "SalesLag_1" in lags
        assert "RollingMean_7" in lags
        assert "RollingStd_7" in lags

        # Проверка, что данные загружаются из БД при новом экземпляре
        new_manager = SalesHistoryManager(
            history_file = str(self.history_file),
            db_path = self.db_path
        )
        assert len(new_manager.history) == 3
        assert new_manager.get_history_stats()["total_records"] == 3

    def test_history_update_with_duplicates(self):
        """Тест обновления истории с дублирующимися записями (upsert в БД)"""
        initial_data = pd.DataFrame({
            "Store": [1, 1],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Sales": [1000.0, 1200.0],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        update_data = pd.DataFrame({
            "Store": [1, 2], # Одна обновляемая запись, одна новая
            "Date": pd.to_datetime(["2024-01-02", "2024-01-01"]), # Обновление продаж за 02.01
            "Sales": [1500.0, 800.0], # Обновленные продажи
            "DayOfWeek": [2, 1],
            "Promo": [0, 1],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        # Первое обновление
        self.manager.update_history(initial_data)

        # Второе обновление с дубликатами
        self.manager.update_history(update_data)

        # Проверяем, что дубликат обновился, а не добавился
        assert len(self.manager.history) == 3 # 2 для магазина 1 + 1 для магазина 2

        # Проверяем, что продажи обновились
        store_1_data = self.manager.history[self.manager.history["Store"] == 1]
        jan_2_sales = store_1_data[store_1_data["Date"] == pd.Timestamp("2024-01-02")]["Sales"].iloc[0]

        assert jan_2_sales == 1500.0 # Обновленное значение

        # Проверка персистентности в БД
        new_manager = SalesHistoryManager(
            history_file = str(self.history_file),
            db_path = self.db_path
        )

        assert len(new_manager.history) == 3

        store_1_data_new = new_manager.history[new_manager.history["Store"] == 1]
        jan_2_sales_new = store_1_data_new[store_1_data_new["Date"] == pd.Timestamp("2024-01-02")]["Sales"].iloc[0]

        assert jan_2_sales_new == 1500.0

    def test_get_latest_sales(self):
        """Тест получения последних продаж"""
        # Создаем историю с известными данными
        dates = pd.date_range("2024-01-01", periods=10)
        test_data = pd.DataFrame({
            "Store": [1] * 10,
            "Date": dates,
            "Sales": [1000 + i * 100 for i in range(10)],  # Линейный рост
            "DayOfWeek": [(i % 7) + 1 for i in range(10)],
            "Promo": [1 if i % 3 == 0 else 0 for i in range(10)],
            "StateHoliday": ["0"] * 10,
            "SchoolHoliday": [0] * 10
        })

        self.manager.update_history(test_data)

        # Получаем последние 5 продаж
        latest_sales = self.manager.get_latest_sales(1, days=5)

        assert latest_sales is not None
        assert len(latest_sales) <= 5

        # Проверяем, что это действительно последние продажи перед максимальной датой
        max_date = self.manager.history["Date"].max()
        assert all(latest_sales.index < max_date)

    def test_cleanup_old_data(self):
        """Тест очистки устаревших данных c проверкой сохранения в БД"""
        # Создаем данные с разными датами
        old_data = pd.DataFrame({
            "Store": [1, 1],
            "Date": pd.to_datetime(["2023-06-01", "2023-07-01"]),
            "Sales": [1000.0, 1200.0],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        recent_data = pd.DataFrame({
            "Store": [1, 2],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Sales": [1500.0, 800.0],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        # Добавляем все данные
        self.manager.update_history(old_data)
        self.manager.update_history(recent_data)

        initial_count = len(self.manager.history)

        # Очищаем данные старше 01.01.2024
        removed_count = self.manager.cleanup_old_data("2024-01-01")

        assert removed_count == 2 # Две старые записи
        assert len(self.manager.history) == initial_count - removed_count

        # Проверяем, что остались только новые данные
        remaining_dates = self.manager.history["Date"]
        assert all(remaining_dates >= pd.Timestamp("2024-01-01"))

        # Проверка, что изменения сохранения в БД
        new_manager = SalesHistoryManager(
            history_file = str(self.history_file),
            db_path = self.db_path
        )

        assert len(new_manager.history) == initial_count - removed_count
        assert (pd.to_datetime(new_manager.history["Date"]) >= pd.Timestamp("2024-01-01")).all()

    def test_export_history(self):
        """Тест экспорта исторических данных"""
        test_data = pd.DataFrame({
            "Store": [1, 2],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Sales": [1000.0, 1200.0],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        self.manager.update_history(test_data)

        # Экспортируем данные
        export_path = self.test_dir / "exported_history.csv"
        success = self.manager.export_history(export_path)

        assert success
        assert export_path.exists()

        # Проверяем, что файл можно загрузить
        exported_data = pd.read_csv(export_path, parse_dates=["Date"])
        assert len(exported_data) == 2
        assert "Sales" in exported_data.columns
        assert exported_data["Sales"].sum() == 2200.0

class TestEdgeCases:
    """Тесты граничных случаев"""

    def test_feature_engineering_empty_data(self):
        """Тест feature engineering с пустыми данными"""
        engineer = FeatureEngineer()

        empty_data = pd.DataFrame(columns=[
            "Store", "Date", "Sales", "DayOfWeek", "Promo", "StateHoliday",
            "SchoolHoliday", "CompetitionDistance", "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear",
            "PromoInterval", "StoreType", "Assortment"
        ])

        empty_data["Date"] = pd.to_datetime(empty_data["Date"])

        # Должен корректно обработать пустые данные
        result, features = engineer.prepare_final_dataset(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(features, list)
        assert len(result) == 0

    def test_preprocessor_empty_data(self):
        """Тест предобработки с пустыми данными"""
        preprocessor = DataPreprocessor()

        empty_data = pd.DataFrame(columns=[
            "Store", "Date", "Sales", "Open", "DayOfWeek",
            "Promo", "StateHoliday", "SchoolHoliday"
        ])

        # Должен корректно обработать пустые данные
        result = preprocessor.clean_data(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_history_manager_empty_history(self):
        """Тест менеджера истории с пустой базой (новая БД)"""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Путь к новой БД, которая еще не создавалась
            test_dir = Path(tmpdir)
            empty_db_path = test_dir / "empty.db"
            manager = SalesHistoryManager(
                history_file = str(test_dir / "empty.pkl"),
                db_path = empty_db_path
            )

            # Проверяем статистику пустой истории
            stats = manager.get_history_stats()

            assert stats["total_records"] == 0
            assert stats["unique_stores"] == 0
            assert stats["date_range"] == "База данных пуста"

            # Дополнительно проверяем, что БД создалась
            assert empty_db_path.exists()

    def test_invalid_data_types(self):
        """Тест обработки невалидных типов данных"""
        preprocessor = DataPreprocessor()

        # Данные с неправильными типами
        test_data = pd.DataFrame({
            "Store": ["invalid", "also_invalid"],  # Должны быть числа
            "Date": ["not-a-date", "2024-01-02"],
            "Sales": ["text", "more_text"],  # Должны быть числа
            "Open": [1, 1],
            "DayOfWeek": [1, 2],
            "Promo": [1, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })

        test_data["Store"] = pd.to_numeric(test_data["Store"], errors="coerce")
        test_data["Sales"] = pd.to_numeric(test_data["Sales"], errors="coerce")
        test_data["Date"] = pd.to_datetime(test_data["Date"], errors="coerce")

        # Должен обработать с ошибками преобразования
        result = preprocessor.clean_data(test_data)

        # Проверяем что результат - DateFrame
        assert isinstance(result, pd.DataFrame)

def run_basic_tests():
    """Запуск базовых тестов"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info("Запуск тестов системы...")

    exit_code = pytest.main([__file__, "-v", "--tb=short", "-q"])

    if exit_code == 0:
        logger.info("Все тесты пройдены успешно")
    else:
        logger.error(f"Тесты завершились с кодом: {exit_code}")

    return exit_code

if __name__ == "__main__":
    # При прямом запуске - используем pytest
    exit_code = run_basic_tests()
    exit(exit_code)
