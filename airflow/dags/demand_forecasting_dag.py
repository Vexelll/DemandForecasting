import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from src.data.data_quality_checker import DataQualityChecker
from src.pipeline.pipeline_operations import PipelineOperations

dags_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dags_dir)
from monitoring.dag_monitor import DAGMonitor

class DemandForecastingPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._quality_checker = None
        self._operations = None
        self._monitor = None
        self.default_args = {
            "owner": "data_engineering",
            "depends_on_past": False,
            "start_date": datetime(2024, 1, 1),
            "email_on_failure": True,
            "email_on_retry": False,
            "retries": 2,
            "retry_delay": timedelta(minutes=5),
            "max_active_runs": 1
        }
        self.cleaned_data = None

    @property
    def quality_checker(self):
        """Ленивая инициализация DataQualityChecker"""
        if self._quality_checker is None:
            self._quality_checker = DataQualityChecker()
        return self._quality_checker

    @property
    def operations(self):
        """Ленивая инициализация PipelineOperations"""
        if self._operations is None:
            self._operations = PipelineOperations()
        return self._operations

    @property
    def monitor(self):
        """Ленивая инициализация DAGMonitor"""
        if self._monitor is None:
            self._monitor = DAGMonitor()
        return self._monitor

    def check_new_data(self) -> str:
        """Проверка наличия новых данных"""
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

    def decide_retraining_path(self, **context) -> str:
        """Принятие решения об переобучении модели"""
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
        """Предобработка входных данных"""
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
        """Создание признаков (feature engineering)"""
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
        """Обновление истории продаж"""
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
        """Обучение модели LightGBM"""
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
        """Полное переобучение модели (еженедельное)"""
        self.logger.info("Полное переобучение модели...")
        return self.retrain_model()

    def generate_predictions(self) -> str:
        """Генерация прогнозов через ETL-пайплайн"""
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
        """Валидация результатов прогнозирования"""
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
        """callback обработки ошибок DAG"""
        error = context.get("exception")
        task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown"

        self.logger.error(f"Ошибка в задаче {task_id}: {error}")

    def create_dag(self) -> DAG:
        """Создание основного DAG ежедневного прогнозирования"""
        with DAG(
            "demand_forecasting_pipeline",
            default_args=self.default_args,
            description="Автоматизированный пайплайн прогнозирования спроса",
            schedule="0 2 * * *", # Ежедневно в 2:00
            max_active_runs=1,
            max_active_tasks=1,
            catchup=False,
            tags=["retail", "forecasting"],
            on_failure_callback=self.on_failure_callback
        ) as dag:

            # Старт
            start = EmptyOperator(task_id="start")

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
                execution_timeout=timedelta(hours=3)
            )

            # Пропуски
            skip_processing = EmptyOperator(task_id="skip_processing")
            skip_retraining = EmptyOperator(task_id="skip_retraining")

            # Прогнозирование
            generate_forecasts = PythonOperator(
                task_id="generate_predictions",
                python_callable=self.generate_predictions,
                trigger_rule="none_failed",
                execution_timeout=timedelta(minutes=30),

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

            end = EmptyOperator(task_id="end")

            # Определение зависимостей
            start >> check_data

            check_data >> [preprocess, skip_processing]

            preprocess >> [update_features, update_history]
            [update_features, update_history] >> decide_retraining

            decide_retraining >> [retrain, skip_retraining]
            [retrain, skip_retraining] >> generate_forecasts

            generate_forecasts >> validate >> success_notification >> end

            skip_processing >> end

            return dag

    def create_retraining_dag(self) -> DAG:
        """Создание DAG еженедельного переобучения"""
        with DAG(
            "weekly_model_retraining",
            default_args=self.default_args,
            description="Еженедельное переобучение модели",
            schedule="0 3 * * 0", # Воскресенье в 3:00
            catchup=False,
            tags=["retraining"]
        ) as dag:

            start = EmptyOperator(task_id="start")

            full_retrain = PythonOperator(
                task_id="full_model_retraining",
                python_callable=self.full_retrain_model
            )

            end = EmptyOperator(task_id="end")

            start >> full_retrain >> end

            return dag

# Создание DAG
pipeline = DemandForecastingPipeline()

# Основной DAG
dag = pipeline.create_dag()

# DAG для переобучения
weekly_retraining_dag = pipeline.create_retraining_dag()
