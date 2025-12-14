import json
import re
from datetime import datetime
import logging
from config.settings import DATA_PATH, REPORTS_PATH, MODELS_PATH

class DAGMonitor:
    """Мониторинг выполнения DAG для системы прогнозирования спроса"""

    def __init__(self):
        self.monitoring_file = REPORTS_PATH / "dag_monitoring.json"
        self.logger = logging.getLogger("dag_monitor")
        self.setup_monitoring()

    def setup_monitoring(self):
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

    def log_dag_run(self, dag_name, status, tasks_executed, error=None):
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

        # Сохраняем только последние 50 запусков
        if len(monitoring_data["dag_runs"]) > 50:
            monitoring_data["dag_runs"] = monitoring_data["dag_runs"][-50:]

        self._update_success_metrics(monitoring_data)
        self._check_for_alerts(monitoring_data, run_info)
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

    def generate_performance_report(self):
        """Генерация отчета о производительности DAG"""
        monitoring_data = self._load_monitoring_data()

        # Базовая структура очета с значениями по умолчанию
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

    def _save_performance_report(self, report):
        """Сохранение отчета о производительности"""
        try:
            report_path = REPORTS_PATH / "performance_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения отчета: {e}")

    def _calculate_duration(self):
        """Расчет длительности выполнения"""
        return 120 # секунды

    def _calculate_average_duration(self, runs):
        """Расчет средней длительности выполнения"""
        if not runs:
            return 0
        durations = [r.get("duration_seconds", 0) for r in runs]
        return sum(durations) / len(durations)

    def _update_success_metrics(self, monitoring_data):
        """Обновление метрик успешности"""
        if monitoring_data["dag_runs"]:
            successful_runs_count = len([r for r in monitoring_data["dag_runs"] if r["status"] == "success"])

            monitoring_data["success_rate"] = (successful_runs_count / len(monitoring_data["dag_runs"])) * 100

            # Обновляем время последнего успешного запуска
            successful_runs_list = [r for r in monitoring_data["dag_runs"] if r["status"] == "success"]
            if successful_runs_list:
                monitoring_data["last_success"] = successful_runs_list[-1]["timestamp"]

    def _check_for_alerts(self, monitoring_data):
        """Проверка условий для алертов"""
        alerts = []

        # Алерт при последовательных неудачах
        recent_failures = [r for r in monitoring_data["dag_runs"][-3:] if r["status"] == "failed"]
        if len(recent_failures) >= 2:
            alerts.append("Критично: 2 последовательных неудачных запуска DAG")

        # Алерт при низком проценте успешных запусков
        if monitoring_data.get("success_rate", 0) < 80:
            alerts.append("Низкий процент успешных запусков DAG")

        monitoring_data["alerts"].extend(alerts)

        if alerts:
            self.logger.warning(f"Сгенерированы алерты: {alerts}")

    def _load_monitoring_data(self):
        """Загрузка данных мониторинга"""
        try:
            with open(self.monitoring_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"dag_runs": [], "success_rate": 0, "last_success": None, "metrics_trend": [], "alerts": []}

    def _save_monitoring_data(self, data):
        """Сохранение данных мониторинга"""
        try:
            with open(self.monitoring_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных мониторинга: {e}")

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
                freshness_str = data_quality["data_freshness"]

                total_hours = 0

                if "day" in freshness_str:
                    parts = freshness_str.split(",")
                    if len(parts) == 2:
                        days_part = parts[0].strip()
                        time_part = parts[1].strip()

                        # Извлекаем количество дней
                        days_match = re.search(r"(\d+)\s+day", days_part)
                        if days_match:
                            days = int(days_match.group(1))
                            total_hours += days * 24

                        # Извлекаем часы из временной части
                        time_match = re.match(r"(\d+):(\d+):(\d+)", time_part)
                        if time_match:
                            hours = int(time_match.group(1))
                            total_hours += hours
                else:
                    time_match = re.match(r"(\d+):(\d+):(\d+)", freshness_str)
                    if time_match:
                        hours = int(time_match.group(1))
                        total_hours = hours

                # Данные считаются свежими, если им меньше 24 часов
                if total_hours < 24:
                    health_score += 25
                else:
                    self.logger.warning(f"Данные устарели: {total_hours} часов")

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
    print("Проверка состояния системы прогнозирования спроса...")

    monitor = DAGMonitor()

    # Проверка качества данных
    print("\n1. Проверка качества данных:")
    data_quality = monitor.check_data_quality()
    for check, result in data_quality.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} {check}: {result}")

    # Генерация отчета
    print("\n2. Отчет о производительности:")
    report = monitor.generate_performance_report()

    print(f"Всего запусков: {report["total_runs"]}")
    print(f"Успешных запусков: {report["successful_runs"]}")
    print(f"Процент успеха: {report["success_rate"]:.1f}%")
    print(f"Состояние системы: {report["system_health"]["status"]}")

    # Показ алертов
    monitoring_data = monitor._load_monitoring_data()
    if monitoring_data.get("alerts"):
        print("\n3. Активные алерты:")
        for alert in monitoring_data["alerts"][-5:]:
            print(f"ALERT: {alert}")

    print(f"\nПодробный отчет сохранен: reports/performance_report.json")


if __name__ == "__main__":
    main()