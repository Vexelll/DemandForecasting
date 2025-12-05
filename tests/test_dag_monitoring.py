import unittest
import tempfile
import json
import shutil
from pathlib import Path
from airflow.dags.monitoring.dag_monitor import DAGMonitor


class TestDAGMonitor(unittest.TestCase):
    """Тесты для мониторинга DAG"""

    def setUp(self):
        """Инициализация тестового окружения"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reports_path = self.temp_dir / "reports"
        self.reports_path.mkdir()

    def tearDown(self):
        """Очистка тестового окружения"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_monitor_initialization(self):
        """Тест инициализации монитора"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.setup_monitoring()

        self.assertTrue(monitor.monitoring_file.exists())

        # Проверяем структуру созданного файла
        with open(monitor.monitoring_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_keys = ["dag_runs", "success_rate", "last_success", "metrics_trend", "alerts"]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_monitor_initialization_existing_file(self):
        """Тест инициализации с существующим файлом"""
        # Создаем файл с данными заранее
        monitoring_file = self.reports_path / "test_monitoring.json"
        existing_data = {
            "dag_runs": [{"test": "data"}],
            "success_rate": 85.5,
            "last_success": "2024-01-01T00:00:00",
            "metrics_trend": [],
            "alerts": []
        }

        with open(monitoring_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        monitor = DAGMonitor()
        monitor.monitoring_file = monitoring_file
        monitor.setup_monitoring()

        # Проверяем что существующие данные не перезаписаны
        with open(monitoring_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["success_rate"], 85.5)

    def test_dag_run_logging_success(self):
        """Тест логирования успешного запуска DAG"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.setup_monitoring()

        monitor.log_dag_run("test_dag", "success", 5)

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["dag_runs"][0]["dag_name"], "test_dag")
        self.assertEqual(data["dag_runs"][0]["status"], "success")
        self.assertEqual(data["dag_runs"][0]["tasks_executed"], 5)
        self.assertIsNone(data["dag_runs"][0]["error"])

    def test_dag_run_logging_failure(self):
        """Тест логирования неудачного запуска DAG"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.setup_monitoring()

        monitor.log_dag_run("test_dag", "failed", 3, "File not found")

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["dag_runs"][0]["status"], "failed")
        self.assertEqual(data["dag_runs"][0]["tasks_executed"], 3)
        self.assertEqual(data["dag_runs"][0]["error"], "File not found")

    def test_multiple_dag_runs_logging(self):
        """Тест логирования нескольких запусков DAG"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.setup_monitoring()

        # Логируем несколько запусков
        for i in range(3):
            monitor.log_dag_run(
                f"test_dag_{i}",
                "success" if i % 2 == 0 else "failed",
                i + 1,
                None if i % 2 == 0 else f"Error {i}"
            )

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 3)

        # Проверяем метрики успешности
        self.assertIn("success_rate", data)
        self.assertGreaterEqual(data["success_rate"], 0)
        self.assertLessEqual(data["success_rate"], 100)

    def test_data_quality_check_basic(self):
        """Базовый тест проверки качества данных"""
        monitor = DAGMonitor()
        checks = monitor.check_data_quality()

        required_checks = ["training_data_exists", "model_exists", "predictions_exist", "data_freshness"]

        for check in required_checks:
            self.assertIn(check, checks)

    def test_generate_performance_report(self):
        """Тест генерации отчета о производительности"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.setup_monitoring()

        # Добавляем тестовые данные
        monitor.log_dag_run("test_dag", "success", 5)
        monitor.log_dag_run("test_dag", "failed", 3, "Test error")

        report = monitor.generate_performance_report()

        required_report_keys = [
            "report_generated", "total_runs", "successful_runs",
            "success_rate", "data_quality", "system_health"
        ]

        for key in required_report_keys:
            self.assertIn(key, report)

        self.assertEqual(report["total_runs"], 2)
        self.assertEqual(report["successful_runs"], 1)
        self.assertEqual(report["success_rate"], 50.0)

    def test_system_health_assessment(self):
        """Тест оценки состояния системы"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"

        health = monitor._assess_system_health()

        self.assertIn("health_score", health)
        self.assertIn("status", health)
        self.assertIn(health["status"], ["healthy", "degraded", "critical"])


if __name__ == "__main__":
    unittest.main()