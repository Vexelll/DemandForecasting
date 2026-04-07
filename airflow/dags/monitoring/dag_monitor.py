import logging
import json
from datetime import datetime
from typing import Any

from config.settings import MODELS_PATH, REPORTS_PATH, resolve_data_path, setup_logging, get_model_config, get_monitoring_config, get_reporting_config

class DAGMonitor:
    """Трекает запуски DAG, считает success rate, генерит алерты"""

    def __init__(self):
        monitoring_cfg = get_monitoring_config()
        self.MAX_STORED_RUNS = monitoring_cfg.get("max_stored_runs", 50)
        self.DATA_FRESHNESS_THRESHOLD_HOURS = monitoring_cfg.get("data_freshness_threshold_hours", 24)
        self.SUCCESS_RATE_ALERT_THRESHOLD = monitoring_cfg.get("success_rate_alert_threshold", 80)
        self.CONSECUTIVE_FAILURES_ALERT = monitoring_cfg.get("consecutive_failures_alert", 2)

        report_files = get_reporting_config().get("output_files", {})
        self.monitoring_file = REPORTS_PATH / report_files.get("dag_monitoring", "dag_monitoring.json")
        self.performance_report_file = REPORTS_PATH / report_files.get("performance_report", "performance_report.json")

        self.logger = logging.getLogger(__name__)
        self._run_start_time = None
        self.setup_monitoring()

    def start_timer(self) -> None:
        """Старт таймера - вызывается в начале DAG"""
        self._run_start_time = datetime.now()

    def setup_monitoring(self) -> None:
        """Создает dag_monitoring.json, если его нет"""
        if not self.monitoring_file.exists():
            base_monitoring = {
                "dag_runs": [],
                "success_rate": 0,
                "last_success": None,
                "metrics_trend": [],
                "alerts": []
            }
            self._save_monitoring_data(base_monitoring)
            self.logger.info("Система мониторинга DAG инициализирована")

    def log_dag_run(self, dag_name: str, status: str, tasks_executed: int, error: str | None = None) -> None:
        """Записывает результат запуска DAG в json"""
        monitoring_data = self._load_monitoring_data()

        run_info = {
            "timestamp": datetime.now().isoformat(),
            "dag_name": dag_name,
            "status": status,
            "tasks_executed": tasks_executed,
            "error": error,
            "duration_seconds": self._calculate_duration()
        }

        monitoring_data["dag_runs"].append(run_info)

        # Ротация: храним только последние N запусков
        if len(monitoring_data["dag_runs"]) > self.MAX_STORED_RUNS:
            monitoring_data["dag_runs"] = monitoring_data["dag_runs"][-self.MAX_STORED_RUNS:]

        self._update_success_metrics(monitoring_data)
        self._check_for_alerts(monitoring_data)
        self._save_monitoring_data(monitoring_data)

        self.logger.info(f"Зафиксирован запуск DAG: {dag_name} - {status}")

    def check_data_quality(self) -> dict[str, Any]:
        """Файлы на месте? Модель есть? Данные свежие? Возраст в часах"""
        checks = {
            "training_data_exists": False,
            "model_exists": False,
            "predictions_exist": False,
            "data_freshness_hours": None
        }

        try:
            train_path = resolve_data_path("raw", "train")
            predictions_path = resolve_data_path("outputs", "predictions")
            model_path = MODELS_PATH / get_model_config().get("model_filename", "lgbm_final_model.pkl")

            checks["training_data_exists"] = train_path.exists()
            checks["model_exists"] = model_path.exists()
            checks["predictions_exist"] = predictions_path.exists()

            if checks["training_data_exists"]:
                data_mtime = datetime.fromtimestamp(train_path.stat().st_mtime)
                data_age_hours = (datetime.now() - data_mtime).total_seconds() / 3600
                checks["data_freshness_hours"] = round(data_age_hours, 1)

            self.logger.info("Проверка качества данных завершена")
            return checks

        except Exception as e:
            self.logger.error(f"Ошибка проверки качества данных: {e}")
            return checks

    def generate_performance_report(self) -> dict[str, Any]:
        """Сводка: total_runs, success rate, failures, health score"""
        monitoring_data = self._load_monitoring_data()

        data_quality = self.check_data_quality()

        report = {
            "report_generated": datetime.now().isoformat(),
            "total_runs": 0,
            "successful_runs": 0,
            "success_rate": 0,
            "average_duration": 0,
            "recent_failures": [],
            "data_quality": data_quality,
            "system_health": self._assess_system_health(data_quality)
        }

        if not monitoring_data["dag_runs"]:
            report["error"] = "Нет данных о запусках DAG"
            self._save_performance_report(report)
            self.logger.warning("Отчет сгенерирован: нет данных о запусках DAG")
            return report

        recent_runs = monitoring_data["dag_runs"][-10:]

        report.update({
            "total_runs": len(monitoring_data["dag_runs"]),
            "successful_runs": len([r for r in monitoring_data["dag_runs"] if r["status"] == "success"]),
            "success_rate": monitoring_data.get("success_rate", 0),
            "average_duration": self._calculate_average_duration(recent_runs),
            "recent_failures": [r for r in recent_runs if r["status"] == "failed"]
        })

        self._save_performance_report(report)
        self.logger.info("Отчет о производительности сгенерирован")

        return report

    def _save_performance_report(self, report: dict[str, Any]) -> None:
        """Дамп отчета в reports/performance_report.json"""
        try:
            with open(self.performance_report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Отчет сохранен: {self.performance_report_file}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения отчета: {e}")

    def _calculate_duration(self) -> float:
        """Секунды с момента start_timer()"""
        if self._run_start_time is not None:
            return (datetime.now() - self._run_start_time).total_seconds()
        return 0.0

    def _calculate_average_duration(self, runs: list[dict[str, Any]]) -> float:
        """Среднее duration_seconds по списку запусков"""
        if not runs:
            return 0.0
        durations = [r.get("duration_seconds", 0) for r in runs]
        return sum(durations) / len(durations)

    def _update_success_metrics(self, monitoring_data: dict[str, Any]) -> None:
        """Пересчет success_rate и last_success за один проход"""
        if not monitoring_data["dag_runs"]:
            return

        successful_count = 0
        last_success_ts = None

        for run in monitoring_data["dag_runs"]:
            if run["status"] == "success":
                successful_count += 1
                last_success_ts = run["timestamp"]

        monitoring_data["success_rate"] = (successful_count / len(monitoring_data["dag_runs"])) * 100
        if last_success_ts:
            monitoring_data["last_success"] = last_success_ts

    def _check_for_alerts(self, monitoring_data: dict[str, Any]) -> None:
        """Алерты: N подряд failures, success rate ниже порога"""
        alerts = []

        # Считаем подряд идущие failures c конца
        consecutive_failures = 0
        for run in reversed(monitoring_data["dag_runs"]):
            if run["status"] == "failed":
                consecutive_failures += 1
            else:
                break

        if consecutive_failures >= self.CONSECUTIVE_FAILURES_ALERT:
            alerts.append(f"Критично: {consecutive_failures} последовательных неудачных запуска DAG")

        success_rate = monitoring_data.get("success_rate", 0)
        if success_rate < self.SUCCESS_RATE_ALERT_THRESHOLD:
            alerts.append(f"Низкий процент успешных запусков DAG: {success_rate:.1f}%")

        monitoring_data["alerts"] = alerts

        if alerts:
            self.logger.warning(f"Сгенерированы алерты: {alerts}")

    def _load_monitoring_data(self) -> dict[str, Any]:
        """json -> dict"""
        default = {"dag_runs": [], "success_rate": 0, "last_success": None, "metrics_trend": [], "alerts": []}
        try:
            with open(self.monitoring_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError as e:
            self.logger.warning(f"Поврежден {self.monitoring_file.name}, история мониторинга сброшена: {e}")
            return default

    def _save_monitoring_data(self, data: dict[str, Any]) -> None:
        """dict -> json"""
        try:
            with open(self.monitoring_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных мониторинга: {e}")

    def _assess_system_health(self, data_quality: dict[str, Any]) -> dict[str, Any]:
        """Качество 0-100: файлы(25*3) + свежесть данных (25)"""
        health_score = 0

        if data_quality.get("training_data_exists"):
            health_score += 25
        if data_quality.get("model_exists"):
            health_score += 25
        if data_quality.get("predictions_exist"):
            health_score += 25

        freshness_hours = data_quality.get("data_freshness_hours")
        if freshness_hours is not None:
            if freshness_hours < self.DATA_FRESHNESS_THRESHOLD_HOURS:
                health_score += 25
            else:
                self.logger.warning(f"Данные устарели: {freshness_hours:.0f} часов")

        if health_score >= 75:
            status = "healthy"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"

        return {"health_score": health_score, "status": status}

def main():
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Проверка состояния системы прогнозирования спроса...")

    monitor = DAGMonitor()

    # Проверка качества данных
    logger.info("1. Проверка качества данных:")
    data_quality = monitor.check_data_quality()
    for check, result in data_quality.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{status} {check}: {result}")

    # Генерация отчета
    logger.info("2. Отчет о производительности:")
    report = monitor.generate_performance_report()

    logger.info(f"Всего запусков: {report["total_runs"]}")
    logger.info(f"Успешных запусков: {report["successful_runs"]}")
    logger.info(f"Процент успеха: {report["success_rate"]:.1f}%")
    logger.info(f"Состояние системы: {report["system_health"]["status"]}")

    # Показ алертов
    monitoring_data = monitor._load_monitoring_data()
    if monitoring_data.get("alerts"):
        logger.info("3. Активные алерты:")
        for alert in monitoring_data["alerts"][-5:]:
            logger.warning(f"ALERT: {alert}")

    logger.info(f"Подробный отчет сохранен: reports/performance_report.json")


if __name__ == "__main__":
    main()
