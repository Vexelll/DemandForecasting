import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from config.settings import DATA_PATH, REPORTS_PATH, MODELS_PATH

class DAGMonitor:
    """Мониторинг выполнения DAG для системы прогнозирования спроса"""
    # Максимальное количество хранимых запусков
    MAX_STORED_RUNS = 50
    # Порог свежести данных (часы)
    DATA_FRESHNESS_THRESHOLD_HOURS = 24
    # Порог success rate для алерта (%)
    SUCCESS_RATE_ALERT_THRESHOLD = 80
    # Количество последовательных неудач для критического алерта
    CONSECUTIVE_FAILURES_ALERT = 2


    def __init__(self):
        self.monitoring_file = REPORTS_PATH / "dag_monitoring.json"
        self.logger = logging.getLogger(__name__)
        self._run_start_time = None
        self.setup_monitoring()

    def start_timer(self) -> None:
        """Начало отсчета времени выполнения DAG"""
        self._run_start_time = datetime.now()

    def setup_monitoring(self) -> None:
        """Инициализация системы мониторинга"""
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

    def log_dag_run(self, dag_name: str, status: str, tasks_executed: int, error: Optional[str] = None) -> None:
        """Логирование выполнения DAG"""
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

        # Сохраняем только последние MAX_STORED_RUNS запусков
        if len(monitoring_data["dag_runs"]) > self.MAX_STORED_RUNS:
            monitoring_data["dag_runs"] = monitoring_data["dag_runs"][-self.MAX_STORED_RUNS:]

        self._update_success_metrics(monitoring_data)
        self._check_for_alerts(monitoring_data)
        self._save_monitoring_data(monitoring_data)

        self.logger.info(f"Зафиксирован запуск DAG: {dag_name} - {status}")

    def check_data_quality(self):
        """Проверка качества данных для DAG"""
        checks = {
            "training_data_exists": False,
            "model_exists": False,
            "predictions_exist": False,
            "data_freshness": None
        }

        try:
            # Проверка существования ключевых файлов
            checks["training_data_exists"] = (DATA_PATH / "raw/train.csv").exists()
            checks["model_exists"] = (MODELS_PATH / "lgbm_final_model.pkl").exists()
            checks["predictions_exist"] = (DATA_PATH / "outputs/predictions.csv").exists()

            # Проверка свежести данных
            if checks["training_data_exists"]:
                data_mtime = (DATA_PATH / "raw/train.csv").stat().st_mtime
                data_age = datetime.now() - datetime.fromtimestamp(data_mtime)
                checks["data_freshness"] = str(data_age)

            self.logger.info("Проверка качества данных завершена")
            return checks

        except Exception as e:
            self.logger.error(f"Ошибка проверки качества данных: {e}")
            return checks

    def generate_performance_report(self) -> Dict[str, Any]:
        """Генерация отчета о производительности DAG"""
        monitoring_data = self._load_monitoring_data()

        # Базовая структура отчета с значениями по умолчанию
        report = {
            "report_generated": datetime.now().isoformat(),
            "total_runs": 0,
            "successful_runs": 0,
            "success_rate": 0,
            "average_duration": 0,
            "recent_failures": [],
            "data_quality": self.check_data_quality(),
            "system_health": self._assess_system_health()
        }

        # Проверяем наличие данных о запусках
        if not monitoring_data["dag_runs"]:
            report["error"] = "Нет данных о запусках DAG"
            # Сохраняем отчет даже при отсутствии данных
            self._save_performance_report(report)
            self.logger.warning("Отчет сгенерирован: нет данных о запусках DAG")
            return report

        recent_runs = monitoring_data["dag_runs"][-10:] # Последние 10 запусков

        # Обновляем отчет реальными данными
        report.update({
            "total_runs": len(monitoring_data["dag_runs"]),
            "successful_runs": len([r for r in monitoring_data["dag_runs"] if r["status"] == "success"]),
            "success_rate": monitoring_data.get("success_rate", 0),
            "average_duration": self._calculate_average_duration(recent_runs),
            "recent_failures": [r for r in recent_runs if r["status"] == "failed"]
        })

        # Сохраняем отчет
        self._save_performance_report(report)
        self.logger.info("Отчет о производительности сгенерирован")

        return report

    def _save_performance_report(self, report: Dict[str, Any]) -> None:
        """Сохранение отчета о производительности в JSON"""
        try:
            report_path = REPORTS_PATH / "performance_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Отчет сохранен: {report_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения отчета: {e}")

    def _calculate_duration(self) -> float:
        """Расчет длительности выполнения (секунды)"""
        if self._run_start_time is not None:
            return (datetime.now() - self._run_start_time).total_seconds()
        return 0.0

    def _calculate_average_duration(self, runs: List[Dict[str, Any]]) -> float:
        """Расчет средней длительности выполнения"""
        if not runs:
            return 0.0
        durations = [r.get("duration_seconds", 0) for r in runs]
        return sum(durations) / len(durations)

    def _update_success_metrics(self, monitoring_data: Dict[str, Any]) -> None:
        """Обновление метрик успешности"""
        if not monitoring_data["dag_runs"]:
            return

        successful_runs_count = len([r for r in monitoring_data["dag_runs"] if r["status"] == "success"])

        monitoring_data["success_rate"] = ((successful_runs_count / len(monitoring_data["dag_runs"])) * 100)

        # Обновляем время последнего успешного запуска
        successful_runs_list = [r for r in monitoring_data["dag_runs"] if r["status"] == "success"]
        if successful_runs_list:
            monitoring_data["last_success"] = successful_runs_list[-1]["timestamp"]

    def _check_for_alerts(self, monitoring_data: Dict[str, Any]) -> None:
        """Проверка условий для генерации алертов"""
        alerts = []

        # Алерт при последовательных неудачах
        recent_runs = monitoring_data["dag_runs"][-3:]
        recent_failures = [r for r in recent_runs if r["status"] == "failed"]
        if len(recent_failures) >= self.CONSECUTIVE_FAILURES_ALERT:
            alerts.append(f"Критично: {len(recent_failures)} последовательных неудачных запуска DAG")

        # Алерт при низком проценте успешных запусков
        success_rate = monitoring_data.get("success_rate", 0)
        if success_rate < self.SUCCESS_RATE_ALERT_THRESHOLD:
            alerts.append(f"Низкий процент успешных запусков DAG: {success_rate:.1f}%")

        monitoring_data["alerts"] = alerts

        if alerts:
            self.logger.warning(f"Сгенерированы алерты: {alerts}")

    def _load_monitoring_data(self) -> Dict[str, Any]:
        """Загрузка данных мониторинга из JSON"""
        try:
            with open(self.monitoring_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"dag_runs": [], "success_rate": 0, "last_success": None, "metrics_trend": [], "alerts": []}

    def _save_monitoring_data(self, data):
        """Сохранение данных мониторинга в JSON"""
        try:
            with open(self.monitoring_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных мониторинга: {e}")

    def _parse_timedelta_hours(self, freshness: str) -> float:
        """Парсинг строкового представления timedelta в часы"""
        total_hours = 0.0

        if "day" in freshness:
            parts = freshness.split(",")
            if len(parts) >= 2:
                days_part = parts[0].strip()
                time_part = parts[1].strip()

                # Извлекаем количество дней
                days_match = re.search(r"(\d+)\s+day", days_part)
                if days_match:
                    total_hours += int(days_match.group(1)) * 24

                # Извлекаем часы из временной части
                time_match = re.match(r"(\d+):(\d+):(\d+)", time_part)
                if time_match:
                    total_hours += int(time_match.group(1))
        else:
            time_match = re.match(r"(\d+):(\d+):(\d+)", freshness)
            if time_match:
                total_hours += float(time_match.group(1))

        return total_hours

    def _assess_system_health(self):
        """Оценка общего состояния системы"""
        data_quality = self.check_data_quality()

        health_score = 0

        # Проверка существования ключевых файлов (по 25 баллов каждый)
        if data_quality.get("training_data_exists"):
            health_score += 25
        if data_quality.get("model_exists"):
            health_score += 25
        if data_quality.get("predictions_exist"):
            health_score += 25

        # Проверка свежести данных
        if data_quality.get("data_freshness"):
            try:
                total_hours = self._parse_timedelta_hours(data_quality["data_freshness"])

                if total_hours < self.DATA_FRESHNESS_THRESHOLD_HOURS:
                    health_score += 25
                else:
                    self.logger.warning(f"Данные устарели: {total_hours:.0f} часов")

            except Exception as e:
                self.logger.error(f"Ошибка анализа свежести данных {data_quality["data_freshness"]}: {e}")

        # Определяем статус системы
        if health_score >= 75:
            status = "healthy"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"

        return {"health_score": health_score, "status": status}

def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
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
