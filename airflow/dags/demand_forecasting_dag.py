import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

dags_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dags_dir)

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.smtp.operators.smtp import EmailOperator

from src.pipeline.pipeline_operations import PipelineOperations
from src.data.data_quality_checker import DataQualityChecker
from monitoring.dag_monitor import DAGMonitor
from config.settings import get_pipeline_config

class DemandForecastingPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._quality_checker = None
        self._operations = None
        self._monitor = None

        pipeline_cfg = get_pipeline_config()

        self.default_args = {
            "owner": pipeline_cfg.get("owner", "data_engineering"),
            "depends_on_past": False,
            "start_date": datetime.strptime(pipeline_cfg.get("start_date", "2024-01-01"), "%Y-%m-%d"),
            "email_on_failure": pipeline_cfg.get("email_on_failure", True),
            "email_on_retry": pipeline_cfg.get("email_on_retry", False),
            "retries": pipeline_cfg.get("retries", 2),
            "retry_delay": timedelta(minutes=pipeline_cfg.get("retry_delay_minutes", 5)),
            "on_failure_callback": self.on_failure_callback
        }

    @property
    def quality_checker(self):
        """Lazy init - создается при первом обращении"""
        if self._quality_checker is None:
            self._quality_checker = DataQualityChecker()
        return self._quality_checker

    @property
    def operations(self):
        """Lazy init - не создаем, если задача не дошла до его шага"""
        if self._operations is None:
            self._operations = PipelineOperations()
        return self._operations

    @property
    def monitor(self):
        """Lazy init - мониторинг DAG"""
        if self._monitor is None:
            self._monitor = DAGMonitor()
        return self._monitor

    def start_monitoring(self) -> None:
        """Первая задача DAG - запуск таймера мониторинга"""
        self.monitor.start_timer()
        self.logger.info("Мониторинг: таймер запущен")

    def log_dag_result(self, **context) -> None:
        """Последняя задача DAG - фиксирует результат запуска + генерит отчет"""
        dag_run = context.get("dag_run")
        dag_id = dag_run.dag_id if dag_run else "unknown"
        run_state = str(dag_run.state) if dag_run else "unknown"

        status = "success" if "running" in run_state.lower() else "failed"
        error_msg = None if status == "success" else f"DAG завершился со статусом: {run_state}"

        self.monitor.log_dag_run(
            dag_name=dag_id,
            status=status,
            tasks_executed=0,
            error=error_msg
        )
        self.monitor.generate_performance_report()

        self.logger.info(f"Мониторинг: {dag_id} -> {status} (state={run_state})")

    def check_new_data(self) -> str:
        """Проверяет mtime train.csv"""
        self.logger.info("Проверка новых данных...")

        try:
            if self.quality_checker.has_new_data():
                self.logger.info("Есть новые данные")
                return "preprocess_data"
            else:
                self.logger.info("Нет новых данных")
                return "skip_processing"
        except Exception as e:
            self.logger.error(f"Ошибка проверки данных: {e}")
            return "preprocess_data"

    def decide_retraining_path(self) -> str:
        """Если модель устарела -> branch на retrain"""
        self.logger.info("Проверка необходимости переобучения...")

        try:
            if self.quality_checker.needs_retraining():
                self.logger.info("Требуется переобучение")
                return "retrain_model"
            else:
                self.logger.info("Переобучение не требуется")
                return "skip_retraining"
        except Exception as e:
            self.logger.error(f"Ошибка проверки модели: {e}")
            return "retrain_model"

    def preprocess_data(self) -> str:
        """Задача: clean + merge"""
        self.logger.info("Запуск предобработки данных...")

        try:
            success = self.operations.preprocess_data()
            status = "success" if success else "failure"
            self.logger.info(f"Предобработка завершена: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Ошибка предобработки: {e}")
            return "failure"

    def update_feature_store(self) -> str:
        """Задача: 60+ признаков"""
        self.logger.info("Создание признаков...")

        try:
            success = self.operations.create_features()
            status = "success" if success else "failure"
            self.logger.info(f"Создание признаков завершено: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Ошибка создания признаков: {e}")
            return "failure"

    def update_sales_history(self) -> str:
        """Задача: update history"""
        self.logger.info("Обновление истории продаж...")

        try:
            success = self.operations.update_sales_history()

            if success:
                self.logger.info("История успешно обновлена")
                return "success"
            else:
                self.logger.error("Не удалось обновить историю")
                return "failure"

        except Exception as e:
            self.logger.error(f"Ошибка при вызове update_sales_history: {e}")
            return "failure"

    def retrain_model(self) -> str:
        """Задача: Optuna + train"""
        self.logger.info("Обучение модели...")

        try:
            success = self.operations.train_model()
            status = "success" if success else "failure"
            self.logger.info(f"Обучение модели завершено: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            return "failure"

    def full_retrain_model(self) -> str:
        """Еженедельно: preprocess -> features -> history -> train"""
        self.logger.info("Полное переобучение модели...")

        steps = [
            ("preprocess_data", self.operations.preprocess_data),
            ("create_features", self.operations.create_features),
            ("update_sales_history", self.operations.update_sales_history),
            ("train_model", self.operations.train_model),
        ]

        for step_name, step_fn in steps:
            try:
                success = step_fn()
                if not success:
                    self.logger.error(f"Шаг {step_name} завершился неудачей")
                    return "failure"
                self.logger.info(f"Шаг {step_name} выполнен")
            except Exception as e:
                self.logger.error(f"Ошибка на шаге {step_name}: {e}")
                return "failure"

        self.logger.info("Полное переобучение завершено")
        return "success"

    def generate_predictions(self) -> str:
        """Задача: ETL predict"""
        self.logger.info("Генерация прогнозов...")

        try:
            success = self.operations.make_predictions()
            status = "success" if success else "failure"
            self.logger.info(f"Генерация прогнозов завершена: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Ошибка генерации прогнозов: {e}")
            return "failure"

    def validate_results(self) -> str:
        """Задача: проверка predictions.csv"""
        self.logger.info("Валидация результатов...")

        try:
            success = self.operations.validate_predictions()
            if success:
                self.logger.info("Валидация пройдена")
                return "success"
            else:
                self.logger.warning("Валидация не пройдена")
                return "warning"
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            return "failure"

    def on_failure_callback(self, context: dict) -> None:
        """on_failure_callback - task-level через default_args"""
        task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown"
        error = context.get("exception")

        self.logger.error(f"Ошибка в задаче {task_id}: {error}")

    def create_dag(self) -> DAG:
        """Ежедневный DAG: monitor -> check data -> predict -> validate -> monitor"""
        pipeline_cfg = get_pipeline_config()
        schedules = pipeline_cfg.get("schedules", {})
        timeouts = pipeline_cfg.get("execution_timeouts", {})

        with DAG(
            pipeline_cfg.get("dag_id", "demand_forecasting_pipeline"),
            default_args=self.default_args,
            description="Автоматизированный пайплайн прогнозирования спроса",
            schedule=schedules.get("daily_forecast", "0 2 * * *"),
            max_active_runs=pipeline_cfg.get("max_active_runs", 1),
            max_active_tasks=1,
            catchup=False,
            tags=["retail", "forecasting"]
        ) as dag:

            start = EmptyOperator(task_id="start")

            # Мониторинг - старт таймера
            start_mon = PythonOperator(
                task_id="start_monitoring",
                python_callable=self.start_monitoring
            )

            # Проверка данных
            check_data = BranchPythonOperator(
                task_id="check_new_data",
                python_callable=self.check_new_data
            )

            # Обработка данных
            preprocess = PythonOperator(
                task_id="preprocess_data",
                python_callable=self.preprocess_data
            )

            update_features = PythonOperator(
                task_id="update_feature_store",
                python_callable=self.update_feature_store
            )

            update_history = PythonOperator(
                task_id="update_sales_history",
                python_callable=self.update_sales_history
            )

            # Решение о переобучении
            decide_retraining = BranchPythonOperator(
                task_id="decide_retraining",
                python_callable=self.decide_retraining_path
            )

            retrain = PythonOperator(
                task_id="retrain_model",
                python_callable=self.retrain_model,
                pool="ml_pool",
                execution_timeout=timedelta(minutes=timeouts.get("retrain_model", 180))
            )

            # Пропуски
            skip_processing = EmptyOperator(task_id="skip_processing")
            skip_retraining = EmptyOperator(task_id="skip_retraining")

            # Прогнозирование
            generate_forecasts = PythonOperator(
                task_id="generate_predictions",
                python_callable=self.generate_predictions,
                trigger_rule="none_failed",
                execution_timeout=timedelta(minutes=timeouts.get("generate_predictions", 30)),

            )

            validate = PythonOperator(
                task_id="validate_results",
                python_callable=self.validate_results
            )

            # Уведомление
            success_notification = EmailOperator(
                task_id="success_notification",
                to=Variable.get("alert_email_recipients", default_var="data_team@company.com"),
                subject="Прогнозы готовы",
                html_content="<h3>Пайплайн выполнен успешно</h3>"
            )

            # Мониторинг - фиксация результата (all_done - срабатывает на любой ветке)
            log_result = PythonOperator(
                task_id="log_dag_result",
                python_callable=self.log_dag_result,
                trigger_rule="all_done"
            )

            end = EmptyOperator(task_id="end")

            # Определение зависимостей
            start >> start_mon >> check_data

            check_data >> [preprocess, skip_processing]

            preprocess >> [update_features, update_history]
            [update_features, update_history] >> decide_retraining

            decide_retraining >> [retrain, skip_retraining]
            [retrain, skip_retraining] >> generate_forecasts

            generate_forecasts >> validate >> success_notification >> log_result >> end

            skip_processing >> log_result

            return dag

    def create_retraining_dag(self) -> DAG:
        """Еженедельный DAG: monitor -> полный retrain -> monitor"""
        pipeline_cfg = get_pipeline_config()
        schedules = get_pipeline_config().get("schedules", {})

        with DAG(
            f"{pipeline_cfg.get("dag_id", "demand_forecasting")}_weekly_retraining",
            default_args=self.default_args,
            description="Еженедельное переобучение модели",
            schedule=schedules.get("weekly_retraining", "0 3 * * 0"),
            max_active_runs=pipeline_cfg.get("max_active_runs", 1),
            catchup=False,
            tags=["retraining"]
        ) as dag:

            start = EmptyOperator(task_id="start")

            start_mon = PythonOperator(
                task_id="start_monitoring",
                python_callable=self.start_monitoring
            )

            full_retrain = PythonOperator(
                task_id="full_model_retraining",
                python_callable=self.full_retrain_model
            )

            log_result = PythonOperator(
                task_id="log_dag_result",
                python_callable=self.log_dag_result,
                trigger_rule="all_done"
            )

            end = EmptyOperator(task_id="end")

            start >> start_mon >> full_retrain >> log_result >> end

            return dag

# Создание DAG
pipeline = DemandForecastingPipeline()

# Основной DAG
dag = pipeline.create_dag()

# DAG для переобучения
weekly_retraining_dag = pipeline.create_retraining_dag()
