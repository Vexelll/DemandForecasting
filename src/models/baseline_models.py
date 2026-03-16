import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config.settings import REPORTS_PATH, all_stores_time_split, get_model_config, get_reporting_config, resolve_data_path, setup_logging
from src.models.base_model import BaseModels

class BaselineModels(BaseModels):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        random_state = get_model_config().get("random_state", 42)
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "MeanBaseline": DummyRegressor(strategy="mean")
        }
        self.trained_models: dict[str, Any] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def _validate_input_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        """Проверяет наличие Sales и числовые типы"""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Все входные данные должны быть предоставлены")

        if len(X_train) != len(y_train):
            raise ValueError(f"Несоответствие размеров X_train ({len(X_train)}) и y_train ({len(y_train)})")

        if len(X_test) != len(y_test):
            raise ValueError(f"Несоответствие размеров X_test ({len(X_test)}) и y_test ({len(y_test)})")

        if len(X_train) < 100:
            self.logger.warning("Малый объем обучающих данных может повлиять на качество моделей")

    def _load_dataset(self, data_path: Path) -> pd.DataFrame:
        """csv -> DataFrame, проверяет что файл не пуст"""
        if not data_path.exists():
            raise FileNotFoundError(f"Файл с данными не найден: {data_path}")

        final_data = pd.read_csv(data_path)

        if final_data.empty:
            raise ValueError("Загруженный файл с данными пуст")

        # Проверка обязательных колонок
        required_columns = ["Sales"]
        missing_columns = [col for col in required_columns if col not in final_data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        return final_data

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, dict[str, Any]]:
        """LinearRegression RandomForest, MeanBaseline - все на одном split"""

        # Валидация входных данных
        self._validate_input_data(X_train, X_test, y_train, y_test)

        self.logger.info("Начало обучения базовых моделей")
        self.logger.info(f"Обучающая выборка: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self.logger.info(f"Тестовая выборка: {X_test.shape[0]} samples")

        results = {}

        for name, model in self.models.items():
            try:
                self.logger.info(f"Обучение модели: {name}")
                start_time = pd.Timestamp.now()

                # Обучение модели
                model.fit(X_train, y_train)
                self.trained_models[name] = model

                # Прогнозирование
                y_pred = model.predict(X_test)

                # Расчет метрик
                metrics = self.calculate_metrics(y_test, y_pred)
                training_time = (pd.Timestamp.now() - start_time).total_seconds()
                metrics["training_time_seconds"] = training_time

                results[name] = metrics

                # Визуализация результатов
                self.plot_predictions(y_test, y_pred, f"{name}_predictions")
                self.plot_residuals(y_test, y_pred, name)

                self.logger.info(f"Модель {name} обучена за {training_time:.2f} секунд. MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}, MAPE: {metrics["MAPE"]:.2f}%")

            except Exception as e:
                self.logger.error(f"Критическая ошибка обучения модели {name}: {e}")
                results[name] = {"error": str(e), "training_time_seconds": 0}

        # Сохранение результатов
        self._save_results(results)
        self._log_summary_report(results)

        return results

    def _save_results(self, results: dict[str, dict[str, Any]]) -> None:
        """results.csv + summary.json + график сравнения"""
        try:
            results_df = pd.DataFrame.from_dict(results, orient="index")

            report_files = get_reporting_config().get("output_files", {})

            # CSV
            results_path = REPORTS_PATH / report_files.get("baseline_results", "baseline_models_results.csv")
            results_df.to_csv(results_path, index=True)

            # JSON - numpy типы -> python native через _to_serializable
            valid_results = {k: v for k, v in results.items() if "error" not in v}
            json_results = {
                "model_comparison": {
                    name: self._to_serializable() for name, metrics in valid_results.items()
                },
                "summary": {
                    "best_model": results_df["MAPE"].idxmin() if "MAPE" in results_df.columns else "N/A",
                    "best_mape": float(results_df["MAPE"].min()) if "MAPE" in results_df.columns else None,
                    "total_training_time": float(results_df["training_time_seconds"].sum()),
                    "execution_timestamp": pd.Timestamp.now().isoformat()
                }
            }

            json_path = REPORTS_PATH / report_files.get("baseline_summary", "baseline_models_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Результаты сохранены: -> {results_path}, сводка -> {json_path}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
            raise

    def _log_summary_report(self, results: dict[str, dict[str, Any]]) -> None:
        """Табличка метрик в лог"""
        self.logger.info("=" * 60)
        self.logger.info("СВОДНЫЙ ОТЧЕТ ПО БАЗОВЫМ МОДЕЛЯМ")
        self.logger.info("=" * 60)

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if valid_results:
            best_model = min(valid_results.items(), key=lambda x: x[1]["MAPE"])
            worst_model = max(valid_results.items(), key=lambda x: x[1]["MAPE"])

            self.logger.info(f"Лучшая модель: {best_model[0]} (MAPE: {best_model[1]["MAPE"]:.2f}%)")
            self.logger.info(f"Худшая модель: {worst_model[0]} (MAPE: {worst_model[1]["MAPE"]:.2f}%)")
            self.logger.info(f"Общее время обучения: {sum(v["training_time_seconds"] for v in valid_results.values()):.2f} секунд")

            # Детализация по всем моделям
            self.logger.info("Детальные результаты")
            for model_name, metrics in valid_results.items():
               self.logger.info(
                   f"{model_name}: MAPE={metrics["MAPE"]:.2f}%, "
                   f"MAE={metrics["MAE"]:.2f}, RMSE={metrics["RMSE"]:.2f}, "
                   f"R²={metrics.get("R2", 0):.4f}, "
                   f"время={metrics.get("training_time_seconds", 0):.2f} сек"
               )
        else:
            self.logger.warning("Нет успешно обученных моделей для анализа")

        # Модели с ошибками
        failed_models = {k: v for k, v in results.items() if "error" in v}
        if failed_models:
            self.logger.warning("Модели с ошибками:")
            for model_name, error_info in failed_models.items():
                self.logger.warning(f"{model_name}: {error_info.get("error", "Неизвестная ошибка")}")

    def run_complete_analysis(self, data_path: Path | None = None, train_time_ratio: float | None = None) -> dict[str, dict[str, Any]]:
        """Загрузка -> обучение всех -> сохранение -> отчет"""
        try:
            if data_path is None:
                data_path = resolve_data_path("processed", "final_dataset")

            if train_time_ratio is None:
                train_time_ratio = get_model_config().get("train_time_ratio", 0.8)

            self.logger.info("Загрузка данных для анализа базовых моделей...")
            final_data = self._load_dataset(data_path)

            self.logger.info("Разделение данных на обучающую и тестовую выборки...")
            X_train, X_test, y_train, y_test = all_stores_time_split(final_data, train_time_ratio)

            self.logger.info("Обучение и оценка базовых моделей...")
            self.results = self.train_and_evaluate(X_train, X_test, y_train, y_test)

            return self.results

        except Exception as e:
            self.logger.error(f"Ошибка выполнения анализа базовых моделей: {e}")
            raise

    def get_best_model(self) -> tuple[Any, str]:
        """Лучшая по mape"""
        if not self.results:
            raise ValueError("Анализ не был выполнен. Сначала запустите run_complete_analysis()")

        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        if not valid_results:
            raise ValueError("Нет успешно обученных моделей")

        best_model_name = min(valid_results.items(), key=lambda x: x[1]["MAPE"])[0]
        return self.trained_models[best_model_name], best_model_name

    def get_model_comparison_data(self) -> pd.DataFrame:
        """metrics dict -> DataFrame для удобного сравнения"""
        if not self.results:
            raise ValueError("Анализ не был выполнен. Сначала запустите run_complete_analysis()")

        return pd.DataFrame.from_dict(self.results, orient="index")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        logger.info("Инициализация анализа базовых моделей...")
        baseline_analyzer = BaselineModels()
        results = baseline_analyzer.run_complete_analysis()

        if results:
            best_model, best_model_name = baseline_analyzer.get_best_model()
            logger.info(f"Анализ базовых моделей успешно завершен")
            logger.info(f"Рекомендуемая модель для дальнейшего использования: {best_model_name}")

            # Сравнительная визуализация - только успешные модели
            valid_results = {k: v for k, v in results.items() if "error" not in v}
            if valid_results:
                baseline_analyzer.plot_metrics_comparison(valid_results, "Сравнение базовых моделей")

        return results

    except Exception as e:
        logger.error(f"Критическая ошибка в анализе базовых моделей: {e}")
        return None

if __name__ == "__main__":
    main()
