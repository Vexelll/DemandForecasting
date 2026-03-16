import json
import shutil
import tempfile
import unittest
from pathlib import Path

from airflow.dags.monitoring.dag_monitor import DAGMonitor
from airflow.sdk.bases.decorator import FParams


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

    def _create_monitor(self) -> DAGMonitor:
        """Вспомогательный метод: создание монитора с тестовым файлом"""
        monitor = DAGMonitor()
        monitor.monitoring_file = self.reports_path / "test_monitoring.json"
        monitor.performance_report_file = self.reports_path / "test_performance.json"
        monitor.setup_monitoring()
        return monitor

    def test_monitor_initialization(self):
        """Тест инициализации монитора с новым файлом"""
        monitor = self._create_monitor()

        self.assertTrue(monitor.monitoring_file.exists())

        with open(monitor.monitoring_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_keys = ["dag_runs", "success_rate", "last_success", "metrics_trend", "alerts"]
        for key in expected_keys:
            self.assertIn(key, data)

        self.assertEqual(data["dag_runs"], [])
        self.assertEqual(data["success_rate"], 0)
        self.assertIsNone(data["last_success"])

    def test_monitor_initialization_existing_file(self):
        """Тест инициализации с существующим файлом - данные не перезаписываются"""
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

        # Проверяем, что существующие данные не перезаписаны
        with open(monitoring_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["success_rate"], 85.5)

    def test_dag_run_logging_success(self):
        """Тест логирования успешного запуска DAG"""
        monitor = self._create_monitor()

        monitor.log_dag_run("test_dag", "success", 5)

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["dag_runs"][0]["dag_name"], "test_dag")
        self.assertEqual(data["dag_runs"][0]["status"], "success")
        self.assertEqual(data["dag_runs"][0]["tasks_executed"], 5)
        self.assertIsNone(data["dag_runs"][0]["error"])
        self.assertIn("timestamp", data["dag_runs"][0])
        self.assertIn("duration_seconds", data["dag_runs"][0])

    def test_dag_run_logging_failure_with_error(self):
        """Тест логирования неудачного запуска DAG с сообщением об ошибке"""
        monitor = self._create_monitor()

        monitor.log_dag_run("test_dag", "failed", 3, "FileNotFoundError: train.csv")

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 1)
        self.assertEqual(data["dag_runs"][0]["status"], "failed")
        self.assertEqual(data["dag_runs"][0]["error"], "FileNotFoundError: train.csv")
        self.assertEqual(data["dag_runs"][0]["tasks_executed"], 3)

    def test_multiple_dag_runs_logging(self):
        """Тест логирования нескольких запусков DAG с правильным success_rate"""
        monitor = self._create_monitor()

        monitor.log_dag_run("dag_1", "success", 5)
        monitor.log_dag_run("dag_2", "failed", 3, "Error")
        monitor.log_dag_run("dag_3", "success", 4)

        data = monitor._load_monitoring_data()

        self.assertEqual(len(data["dag_runs"]), 3)
        self.assertAlmostEqual(data["success_rate"], 66.67, places=1)

    def test_dag_runs_storage_limit(self):
        """Тест ограничения хранения запусков (MAX_STORED_RUNS)"""
        monitor = self._create_monitor()

        # Логируем больше записей, чем лимит
        for i in range(monitor.MAX_STORED_RUNS + 10):
            monitor.log_dag_run(f"dag_{i}", "success", 1)

        data = monitor._load_monitoring_data()

        # Не больше MAX_STORED_RUNS
        self.assertLessEqual(len(data["dag_runs"]), monitor.MAX_STORED_RUNS)

        # Последний запуск - самый свежий
        last_run = data["dag_runs"][-1]
        self.assertEqual(last_run["dag_name"], f"dag_{monitor.MAX_STORED_RUNS + 9}")

    def test_success_rate_calculation(self):
        """Тест корректного расчета success rate"""
        monitor = self._create_monitor()

        monitor.log_dag_run("dag", "success", 5)
        monitor.log_dag_run("dag", "failed", 3)

        data = monitor._load_monitoring_data()

        self.assertAlmostEqual(data["success_rate"], 50.0, places=1)

    def test_last_success_tracking(self):
        """Тест отслеживания последнего успешного запуска"""
        monitor = self._create_monitor()

        monitor.log_dag_run("dag", "failed", 1)
        monitor.log_dag_run("dag", "success", 5)
        monitor.log_dag_run("dag", "failed", 2)

        data = monitor._load_monitoring_data()

        self.assertIsNotNone(data["last_success"])

        # last_success - timestamp второго запуска (единственный success)
        self.assertEqual(data["last_success"], data["dag_runs"][1]["timestamp"])

    def test_alert_on_consecutive_failures(self):
        """Тест: N подряд failures в конце -> алерт"""
        monitor = self._create_monitor()

        # Два подряд неудачных запуска
        monitor.log_dag_run("dag", "failed", 1, "Error 1")
        monitor.log_dag_run("dag", "failed", 2, "Error 2")

        data = monitor._load_monitoring_data()

        # Должен быть хотя бы один алерт о последовательных неудачах
        self.assertTrue(len(data["alerts"]) > 0)
        self.assertTrue(any("Критично" in alert for alert in data["alerts"]))

    def test_no_alert_when_failures_not_consecutive(self):
        """Тест: [fail, success, fail] -> не подряд, алерта о подряд failures нет"""
        monitor = self._create_monitor()

        monitor.log_dag_run("dag", "failed", 1, "Error 1")
        monitor.log_dag_run("dag", "success", 5)
        monitor.log_dag_run("dag", "failed", 2, "Error 2")

        data = monitor._load_monitoring_data()

        # Подряд failures = 1 (только последний), порог = 2 -> нет алерта "Критично"
        critical_alerts = [a for a in data["alerts"] if "Критично" in a]
        self.assertEqual(len(critical_alerts), 0)

    def test_alert_resets_after_success(self):
        "Тест: [fail, fail, success] -> подряд сбрасывается, алерта нет"
        monitor = self._create_monitor()

        monitor.log_dag_run("dag", "failed", 1, "Error 1")
        monitor.log_dag_run("dag", "failed", 2, "Error 2")
        monitor.log_dag_run("dag", "success", 5)

        data = monitor._load_monitoring_data()

        critical_alerts = [a for a in data["alerts"] if "Критично" in a]
        self.assertEqual(len(critical_alerts), 0)

    def test_timer_duration(self):
        monitor = self._create_monitor()

        monitor.start_timer()
        duration = monitor._calculate_duration()

        self.assertGreater(duration, 0)

    def test_duration_without_timer(self):
        "Тест: start_timer() -> _claculate_duration() > 0"
        monitor = self._create_monitor()

        duration = monitor._calculate_duration()

        self.assertEqual(duration, 0.0)

    def test_logged_run_has_duration(self):
        """Тест: после start_timer() -> log_dag_run записывает duration > 0"""
        monitor = self._create_monitor()

        monitor.start_timer()
        monitor.log_dag_run("dag", "success", 5)

        data = monitor._load_monitoring_data()

        self.assertGreater(data["dag_runs"][0]["duration_seconds"], 0)

    def test_data_quality_check_basic(self):
        """Базовый тест проверки качества данных - структура ответа"""
        monitor = DAGMonitor()
        checks = monitor.check_data_quality()

        required_checks = ["training_data_exists", "model_exists", "predictions_exist", "data_freshness_hours"]

        for check in required_checks:
            self.assertIn(check, checks)

    def test_data_quality_freshness_is_numeric(self):
        """Тест: data_freshness_hours - число или None, не строка"""
        monitor = DAGMonitor()
        checks = monitor.check_data_quality()

        freshness = checks["data_freshness_hours"]
        if freshness is not None:
            self.assertIsInstance(freshness, (int, float))

    def test_system_health_assessment_all_missing(self):
        """Тест: все файлы отсутствуют -> health_score = 0, status = critical"""
        monitor = self._create_monitor()

        fake_quality = {
            "training_data_exists": False,
            "model_exists": False,
            "predictions_exist": False,
            "data_freshness_hours": None
        }

        health = monitor._assess_system_health(fake_quality)

        self.assertEqual(health["health_score"], 0)
        self.assertEqual(health["status"], "critical")

    def test_system_health_assessment_all_present_fresh(self):
        """Тест: все файлы есть + данные свежие -> health_score = 100, status = healthy"""
        monitor = self._create_monitor()

        fake_quality = {
            "training_data_exists": True,
            "model_exists": True,
            "predictions_exist": True,
            "data_freshness_hours": 2.0
        }

        health = monitor._assess_system_health(fake_quality)

        self.assertEqual(health["health_score"], 100)
        self.assertEqual(health["status"], "healthy")

    def test_system_health_assessment_stale_data(self):
        """Тест: файлы есть, но данные устаревшие -> 75, healthy"""
        monitor = self._create_monitor()

        fake_quality = {
            "training_data_exists": True,
            "model_exists": True,
            "predictions_exist": True,
            "data_freshness_hours": 999.0
        }

        health = monitor._assess_system_health(fake_quality)

        self.assertEqual(health["health_score"], 75)
        self.assertEqual(health["status"], "healthy")

    def test_system_health_assessment_degraded(self):
        """Тест: 2 из 4 -> 50, degraded"""
        monitor = self._create_monitor()

        fake_quality = {
            "training_data_exists": True,
            "model_exists": True,
            "predictions_exist": False,
            "data_freshness_hours": None
        }

        health = monitor._assess_system_health(fake_quality)

        self.assertEqual(health["health_score"], 50)
        self.assertEqual(health["status"], "degraded")

    def test_generate_performance_report(self):
        """Тест генерации отчета о производительности"""
        monitor = self._create_monitor()

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

    def test_generate_performance_report_empty(self):
        """Тест отчета при пустой истории запусков"""
        monitor = self._create_monitor()

        report = monitor.generate_performance_report()

        self.assertEqual(report["total_runs"], 0)
        self.assertIn("error", report)

    def test_performance_report_saved_to_file(self):
        """Тест: отчет сохраняется в файл"""
        monitor = self._create_monitor()

        monitor.log_dag_run("dag", "success", 5)
        monitor.generate_performance_report()

        self.assertTrue(monitor.performance_report_file.exists())

        with open(monitor.performance_report_file, "r", encoding="utf-8") as f:
            saved_report = json.load(f)

        self.assertIn("total_runs", saved_report)
        self.assertEqual(saved_report["total_runs"], 1)


if __name__ == "__main__":
    unittest.main()
