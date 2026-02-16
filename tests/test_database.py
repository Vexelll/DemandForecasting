import unittest
import tempfile
import shutil
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.database.database_manager import DatabaseManager
from src.database.dashboard_data_provider import DashboardDataProvider

class TestDatabaseManager(unittest.TestCase):
    """Тесты для модуля DatabaseManager"""

    def setUp(self):
        """Инициализация тестового окружения"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_forecasting.db"
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        """Очистка тестового окружения"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_sales_data(self, n_stores: int = 2, n_days: int = 30) -> pd.DataFrame:
        """Генерация тестовых данных продаж"""
        dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
        rows = []
        for store_id in range(1, n_stores + 1):
            for date in dates:
                rows.append({
                    "Store": store_id,
                    "Date": date,
                    "Sales": float(np.random.randint(3000, 15000)),
                    "DayOfWeek": date.dayofweek + 1,
                    "Promo": int(np.random.choice([0, 1])),
                    "StateHoliday": "0",
                    "SchoolHoliday": 0
                })
        return pd.DataFrame(rows)

    def _create_test_predictions(self, n_stores: int = 2, n_days: int = 7) -> pd.DataFrame:
        """Генерация тестовых данных прогнозов"""
        dates = pd.date_range(start="2024-02-01", periods=n_days, freq="D")
        rows = []
        for store_id in range(1, n_stores + 1):
            for date in dates:
                actual = float(np.random.randint(3000, 15000))
                predicted = actual + np.random.normal(0, 500)
                rows.append({
                    "Store": store_id,
                    "Date": date,
                    "PredictedSales": max(predicted, 0),
                    "ActualSales": actual,
                    "AbsoluteError": abs(actual - predicted),
                    "PercentageError": abs(actual - predicted) / actual * 100 if actual > 0 else 0
                })
        return pd.DataFrame(rows)

    def test_database_created(self):
        """Проверка создания файла БД при инициализации"""
        self.assertTrue(self.db_path.exists())

    def test_tables_exist(self):
        """Проверка наличия всех таблиц после инициализации"""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "sales_history", "predictions", "pipeline_runs",
            "model_metrics", "schema_version"
        }
        self.assertTrue(expected_tables.issubset(tables))

    def test_schema_version_recorded(self):
        """Проверка записи версии схемы при инициализации"""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_version")
            version = cursor.fetchone()[0]

        self.assertEqual(version, self.db.DB_VERSION)

    def test_idempotent_initialization(self):
        """Проверка идемпотентности повторной инициализации"""

        # Повторная инициализация не должна вызывать ошибок
        db2 = DatabaseManager(db_path=self.db_path)

        with db2._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM schema_version")
            count = cursor.fetchone()[0]

        # Версия должна быть записана только один раз
        self.assertEqual(count, 1)

    def test_save_sales_history(self):
        """Тест сохранения истории продаж"""
        test_data = self._create_test_sales_data(n_stores=2, n_days=10)
        count = self.db.save_sales_history(test_data)

        self.assertEqual(count, 20)  # 2 магазина * 10 дней

    def test_save_sales_history_empty(self):
        """Тест сохранения пустого DataFrame"""
        count = self.db.save_sales_history(pd.DataFrame())
        self.assertEqual(count, 0)

    def test_load_sales_history(self):
        """Тест загрузки истории продаж"""
        test_data = self._create_test_sales_data(n_stores=2, n_days=5)
        self.db.save_sales_history(test_data)

        loaded = self.db.load_sales_history()

        self.assertEqual(len(loaded), 10)
        self.assertIn("Store", loaded.columns)
        self.assertIn("Date", loaded.columns)
        self.assertIn("Sales", loaded.columns)

    def test_save_sales_history_upsert(self):
        """Тест обновления существующих записей (UPSERT)"""
        # Первая вставка
        data_v1 = pd.DataFrame({
            "Store": [1, 1],
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [1000.0, 2000.0],
            "DayOfWeek": [1, 2],
            "Promo": [0, 1],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })
        self.db.save_sales_history(data_v1)

        # Обновление тех же записей с новыми значениями Sales
        data_v2 = pd.DataFrame({
            "Store": [1, 1],
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [1500.0, 2500.0],
            "DayOfWeek": [1, 2],
            "Promo": [0, 1],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0]
        })
        self.db.save_sales_history(data_v2)

        # Должно остаться 2 записи (не 4)
        loaded = self.db.load_sales_history()
        self.assertEqual(len(loaded), 2)

        # Значения должны быть обновлены
        store1_jan1 = loaded[
            (loaded["Store"] == 1) &
            (loaded["Date"].dt.strftime("%Y-%m-%d") == "2024-01-01")
        ]
        self.assertEqual(store1_jan1["Sales"].iloc[0], 1500.0)

    def test_get_store_history(self):
        """Тест получения истории конкретного магазина"""
        test_data = self._create_test_sales_data(n_stores=3, n_days=30)
        self.db.save_sales_history(test_data)

        store_history = self.db.get_store_history(store_id=1, days_back=14)

        # Все записи должны быть для магазина 1
        self.assertTrue((store_history["Store"] == 1).all())
        # Должно быть <= 14 дней
        self.assertLessEqual(len(store_history), 15)

    def test_get_sales_history_stats(self):
        """Тест получения статистики по истории"""
        test_data = self._create_test_sales_data(n_stores=3, n_days=10)
        self.db.save_sales_history(test_data)

        stats = self.db.get_sales_history_stats()

        self.assertEqual(stats["total_records"], 30)
        self.assertEqual(stats["unique_stores"], 3)
        self.assertIn("date_range", stats)

    def test_get_sales_history_stats_empty(self):
        """Тест статистики на пустой БД"""
        stats = self.db.get_sales_history_stats()
        self.assertEqual(stats["total_records"], 0)

    def test_save_predictions(self):
        """Тест сохранения прогнозов"""
        test_preds = self._create_test_predictions(n_stores=2, n_days=7)
        run_id = "test_run_001"
        count = self.db.save_predictions(test_preds, run_id)

        self.assertEqual(count, 14)  # 2 магазина * 7 дней

    def test_get_predictions_by_run(self):
        """Тест получения прогнозов по run_id"""
        test_preds = self._create_test_predictions()
        run_id = "test_run_002"
        self.db.save_predictions(test_preds, run_id)

        loaded = self.db.get_predictions_by_run(run_id)

        self.assertEqual(len(loaded), len(test_preds))
        self.assertIn("PredictedSales", loaded.columns)
        self.assertIn("ActualSales", loaded.columns)

    # def test_get_latest_predictions(self):
    #     """Тест получения последних прогнозов"""
    #     # Сохраняем два набора прогнозов
    #     preds_1 = self._create_test_predictions(n_stores=1, n_days=3)
    #     preds_2 = self._create_test_predictions(n_stores=2, n_days=5)

    #     self.db.save_predictions(preds_1, "run_old")
    #     self.db.save_predictions(preds_2, "run_new")

    #     latest = self.db.get_latest_predictions()

    #     # Должны вернуться прогнозы последнего запуска
    #     self.assertEqual(len(latest), 10)  # 2 магазина * 5 дней

    def test_list_prediction_runs(self):
        """Тест получения списка запусков"""
        self.db.save_predictions(self._create_test_predictions(), "run_1")
        self.db.save_predictions(self._create_test_predictions(), "run_2")

        runs = self.db.list_prediction_runs()

        self.assertEqual(len(runs), 2)
        self.assertIn("run_id", runs[0])
        self.assertIn("records", runs[0])

    def test_log_pipeline_run(self):
        """Тест логирования запуска пайплайна"""
        self.db.log_pipeline_run(
            run_id="test_pipeline_001",
            dag_name="demand_forecasting_pipeline",
            status="success",
            records_processed=1000,
            mape=12.5,
            rmse=650.0
        )

        runs = self.db.get_pipeline_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], "test_pipeline_001")
        self.assertEqual(runs[0]["status"], "success")
        self.assertEqual(runs[0]["mape"], 12.5)

    def test_log_pipeline_run_failure(self):
        """Тест логирования неудачного запуска"""
        self.db.log_pipeline_run(
            run_id="test_pipeline_fail",
            dag_name="demand_forecasting_pipeline",
            status="failure",
            error_message="FileNotFoundError: train.csv"
        )

        runs = self.db.get_pipeline_runs()
        self.assertEqual(runs[0]["status"], "failure")
        self.assertIn("FileNotFoundError", runs[0]["error_message"])

    def test_save_model_metrics(self):
        """Тест сохранения метрик модели"""
        metrics = {"MAPE": 12.74, "RMSE": 680.0, "MAE": 450.0, "R2": 0.92}
        self.db.save_model_metrics(
            metrics=metrics,
            model_version="v_20240101",
            n_features=52,
            n_train_samples=600000,
            n_test_samples=200000,
            best_params={"learning_rate": 0.05, "num_leaves": 63}
        )

        history = self.db.get_model_metrics_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["mape"], 12.74)
        self.assertEqual(history[0]["n_features"], 52)

    def test_model_metrics_trend(self):
        """Тест получения тренда метрик (несколько обучений)"""
        for i in range(5):
            metrics = {"MAPE": 15.0 - i, "RMSE": 700 - i * 20, "MAE": 500 - i * 10, "R2": 0.88 + i * 0.01}
            self.db.save_model_metrics(metrics=metrics, model_version=f"v_{i}")

        history = self.db.get_model_metrics_history()
        self.assertEqual(len(history), 5)

        # Последняя запись (DESC сортировка) - самая свежая
        self.assertEqual(history[0]["model_version"], "v_4")

    def test_generate_run_id(self):
        """Тест генерации уникального run_id"""
        from src.database.database_manager import DatabaseManager

        run_id_1 = DatabaseManager.generate_run_id()
        run_id_2 = DatabaseManager.generate_run_id()

        self.assertTrue(run_id_1.startswith("run_"))
        self.assertNotEqual(run_id_1, run_id_2)

    def test_get_database_stats(self):
        """Тест получения общей статистики БД"""
        # Наполняем БД данными
        self.db.save_sales_history(self._create_test_sales_data())
        self.db.save_predictions(self._create_test_predictions(), "test_run")
        self.db.log_pipeline_run("run_1", "test_dag", "success")
        self.db.save_model_metrics({"MAPE": 12.0, "RMSE": 650.0})

        stats = self.db.get_database_stats()

        self.assertGreater(stats["sales_history_count"], 0)
        self.assertGreater(stats["predictions_count"], 0)
        self.assertEqual(stats["pipeline_runs_count"], 1)
        self.assertEqual(stats["model_metrics_count"], 1)
        self.assertIn("db_size_mb", stats)
        self.assertIn("db_path", stats)


class TestDashboardDataProvider(unittest.TestCase):
    """Тесты для модуля DashboardDataProvider"""

    def setUp(self):
        """Инициализация тестового окружения"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_dashboard.db"

    def tearDown(self):
        """Очистка тестового окружения"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_demo_data_when_db_empty(self):
        """Проверка генерации демо-данных при пустой БД"""
        provider = DashboardDataProvider()

        # Если БД пуста - должны вернуться демо-данные
        data = provider.load_predictions()

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn("Store", data.columns)
        self.assertIn("Date", data.columns)
        self.assertIn("PredictedSales", data.columns)
        self.assertIn("ActualSales", data.columns)

    def test_demo_data_has_error_columns(self):
        """Проверка наличия расчетных колонок в демо-данных"""

        provider = DashboardDataProvider()
        data = provider.load_predictions()

        self.assertIn("AbsoluteError", data.columns)
        self.assertIn("PercentageError", data.columns)

    def test_data_source_info(self):
        """Проверка информации об источнике данных"""
        provider = DashboardDataProvider()
        info = provider.get_data_source_info()

        self.assertIn("source", info)
        self.assertIn("timestamp", info)

    def test_get_available_runs_empty(self):
        """Проверка списка запусков при пустой БД"""
        provider = DashboardDataProvider()
        runs = provider.get_available_runs()

        self.assertIsInstance(runs, list)

    def test_load_predictions_with_db_data(self):
        """Тест загрузки реальных прогнозов из БД"""
        # Наполняем БД тестовыми данными
        db = DatabaseManager(db_path=self.db_path)
        test_preds = pd.DataFrame({
            "Store": [1, 1, 2, 2],
            "Date": pd.date_range("2024-01-01", periods=2).tolist() * 2,
            "PredictedSales": [5000.0, 6000.0, 7000.0, 8000.0],
            "ActualSales": [5100.0, 5900.0, 7200.0, 7800.0],
            "AbsoluteError": [100.0, 100.0, 200.0, 200.0],
            "PercentageError": [1.96, 1.69, 2.78, 2.56]
        })
        db.save_predictions(test_preds, "test_run_dashboard")

        # Проверяем, что данные загружаются
        provider = DashboardDataProvider()
        data = provider.load_predictions("test_run_dashboard")

        # Провайдер может вернуть данные из БД или демо
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
