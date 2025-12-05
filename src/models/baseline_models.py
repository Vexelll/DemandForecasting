import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from src.models.base_model import BaseModels
from config.settings import DATA_PATH, all_stores_time_split, REPORTS_PATH


class BaselineModels(BaseModels):
    def __init__(self):
        super().__init__()
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "MeanBaseline": DummyRegressor(strategy="mean")
        }
        self.trained_models = {}
        self.results = {}

    def _validate_input_data(self, X_train, X_test, y_train, y_test):
        """Валидация входных данных для обучения"""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Все входные данные должны быть предоставлены")

        if len(X_train) != len(y_train):
            raise ValueError(f"Несоответствие размеров X_train ({len(X_train)}) и y_train ({len(y_train)})")

        if len(X_test) != len(y_test):
            raise ValueError(f"Несоответствие размеров X_test ({len(X_test)}) и y_test ({len(y_test)})")

        if len(X_train) < 100:
            print("Предупреждение: малый объем обучающих данных может повлиять на качество моделей")

    def _load_dataset(self, data_path):
        """Загрузка и валидация набора данных"""
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

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Обучение и оценка базовых моделей"""

        # Валидация входных данных
        self._validate_input_data(X_train, X_test, y_train, y_test)

        print("Начало обучения базовых моделей")
        print(f"Обучающая выборка: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Тестовая выборка: {X_test.shape[0]} samples")

        results = {}
        training_times = {}

        for name, model in self.models.items():
            try:
                print(f"\n--- Обучение модели: {name} ---")
                start_time = pd.Timestamp.now()

                # Обучение модели
                model.fit(X_train, y_train)
                self.trained_models[name] = model

                # Прогнозирование
                y_pred = model.predict(X_test)

                # Расчет метрик
                metrics = self.calculate_metrics(y_test, y_pred)
                training_time = (pd.Timestamp.now() - start_time).total_seconds()
                training_times[name] = training_time
                metrics["training_time_seconds"] = training_time

                results[name] = metrics

                # Визуализация результатов
                self.plot_predictions(y_test, y_pred, f"{name}_predictions")
                self.plot_residuals(y_test, y_pred, name)

                print(f"Модель {name} обучена за {training_time:.2f} секунд")
                print(f"Метрики - MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}, MAPE: {metrics["MAPE"]:.2f}%")

            except Exception as e:
                print(f"Критическая ошибка обучения модели {name}: {e}")
                results[name] = {"error": str(e), "training_time_seconds": 0}

        # Сохранение результатов
        self._save_results(results, training_times)
        self._print_summary_report(results, training_times)

        return results

    def _save_results(self, results, training_times):
        """Сохранение результатов обучения в структурированном виде"""
        try:
            # Преобразование результатов в DataFrame
            results_df = pd.DataFrame.from_dict(results, orient="index")

            # Добавление времени обучения
            results_df["training_time_seconds"] = results_df.index.map(training_times)

            # Сохранение в CSV
            results_path = REPORTS_PATH / "baseline_models_results.csv"
            results_df.to_csv(results_path, index=True)

            # Дополнительное сохранение в JSON для удобства чтения
            json_results = {
                "model_comparison": results_df.to_dict("index"),
                "summary": {
                    "best_model": results_df["MAPE"].idxmin(),
                    "best_mape": results_df["MAPE"].min(),
                    "total_training_time": sum(training_times.values()),
                    "execution_timestamp": pd.Timestamp.now().isoformat()
                }
            }

            json_path = REPORTS_PATH / "baseline_models_summary.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

            print(f"Результаты сохранены:")
            print(f"- Детальные метрики: {results_path}")
            print(f"- Сводный отчет: {json_path}")

        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")
            raise

    def _print_summary_report(self, results, training_times):
        """Вывод сводного отчета по всем моделям"""
        print("СВОДНЫЙ ОТЧЕТ ПО БАЗОВЫМ МОДЕЛЯМ")

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if valid_results:
            best_model = min(valid_results.items(), key=lambda x: x[1]["MAPE"])
            worst_model = max(valid_results.items(), key=lambda x: x[1]["MAPE"])

            print(f"Лучшая модель: {best_model[0]} (MAPE: {best_model[1]["MAPE"]:.2f}%)")
            print(f"Худшая модель: {worst_model[0]} (MAPE: {worst_model[1]["MAPE"]:.2f}%)")
            print(f"Общее время обучения: {sum(training_times.values()):.2f} секунд")

            # Детализация по всем моделям
            print("\nДетальные результаты")
            for model_name, metrics in valid_results.items():
                print(f"- {model_name}: MAPE:{metrics["MAPE"]:.2f}%, "
                      f"MAE={metrics["MAE"]:.2f}, RMSE={metrics["RMSE"]:.2f}, "
                      f"Время={metrics["training_time_seconds"]:.2f}c")
        print("Нет успешно обученных моделей для анализа")

    def run_complete_analysis(self, data_path=None, train_time_ratio=0.8):
        """Полный цикл анализа базовых моделей"""
        try:
            if data_path is None:
                data_path = DATA_PATH / "processed/final_dataset.csv"

            print("Загрузка данных для анализа базовых моделей...")
            final_data = self._load_dataset(data_path)

            print("Разделение данных на обучающую и тестовую выборки...")
            X_train, X_test, y_train, y_test = all_stores_time_split(final_data, train_time_ratio)

            print("Обучение и оценка базовых моделей...")
            self.results = self.train_and_evaluate(X_train, X_test, y_train, y_test)

            return self.results

        except Exception as e:
            print(f"Ошибка выполнения анализа базовых моделей: {e}")
            raise

    def get_best_model(self):
        """Получение лучшей модели на основе метрик"""
        if not self.results:
            raise ValueError("Анализ не был выполнен. Сначала запустите run_complete_analysis()")

        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        if not valid_results:
            raise ValueError("Нет успешно обученных моделей")

        best_model_name = min(valid_results.items(), key=lambda x: x[1]["MAPE"])[0]
        return self.trained_models[best_model_name], best_model_name

    def get_model_comparison_data(self):
        """Получение данных для сравнения моделей в виде DataFrame"""
        if not self.results:
            raise ValueError("Анализ не был выполнен. Сначала запустите run_complete_analysis()")

        return pd.DataFrame.from_dict(self.results, orient="index")

def main():
    """Основная функция для запуска анализа базовых моделей"""
    try:
        print("Инициализация анализа базовых моделей...")
        baseline_analyzer = BaselineModels()
        results = baseline_analyzer.run_complete_analysis()

        if results:
            best_model, best_model_name = baseline_analyzer.get_best_model()
            print(f"\nАнализ базовых моделей успешно завершен")
            print(f"Рекомендуемая модель для дальнейшего использования: {best_model_name}")

            # Создание сравнительной визуализации
            baseline_analyzer.plot_metrics_comparison(results, "Сравнение базовых моделей")

        return results

    except Exception as e:
        print(f"Критическая ошибка в анализе базовых моделей: {e}")
        return None

if __name__ == "__main__":
    main()