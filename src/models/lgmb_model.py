import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.model_selection import TimeSeriesSplit
from src.models.base_model import BaseModels
from config.settings import DATA_PATH, MODELS_PATH, REPORTS_PATH, all_stores_time_split

class LGBMModel(BaseModels):
    def __init__(self):
        super().__init__()
        self.model: Optional[lgb.LGBMRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None
        self.feature_names: List[str] = []
        self.logger = logging.getLogger("lgbm_model")

    def _validate_input_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        """Валидация входных данных для обучения"""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Все входные данные должны быть предоставлены")

        if len(X_train) != len(y_train):
            raise ValueError(f"Несоответствие размеров X_train ({len(X_train)}) и y_train ({len(y_train)})")

        if len(X_test) != len(y_test):
            raise ValueError(f"Несоответствие размеров X_test ({len(X_test)}) и y_test ({len(y_test)})")

        if len(X_train) < 100:
            self.logger.warning("Малый объем обучающих данных может повлиять на качество модели")

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Функция для оптимизации гиперпараметров LightGBM"""
        params = {
            "objective": "regression",
            "metric": "mape",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 5.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 5.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 5.0),
            "random_state": 42
        }

        tscv = TimeSeriesSplit(n_splits=2)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params, n_estimators=1000)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mape",
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(False)
                ]
            )

            y_pred = model.predict(X_val)
            mape = np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100
            scores.append(mape)

        return np.mean(scores)

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 70) -> Dict[str, Any]:
        """Оптимизация гиперпараметров с использованием Optuna"""
        self.logger.info("Начало оптимизации гиперпараметров LightGBM")

        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

        self.study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)

        self.best_params = self.study.best_params
        self.best_params.update({
            "objective": "regression",
            "metric": "mape",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 20000,
            "random_state": 42
        })

        self.logger.info(f"Оптимизация завершена. Лучший MAPE: {self.study.best_value:.2f}%")
        self.logger.info(f"Лучшие параметры: {self.best_params}")

        return self.best_params

    def train_final_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
        """Обучение финальной модели на лучших параметрах"""

        # Валидация входных данных
        self._validate_input_data(X_train, X_test, y_train, y_test)

        self.logger.info("Начало обучения финальной модели LightGBM")

        # Оптимизация гиперпараметров если не выполнена
        if self.best_params is None:
            self.best_params = self.optimize_hyperparameters(X_train, y_train)

        # Обучение финальной модели
        self.model = lgb.LGBMRegressor(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mape",
            callbacks=[
                lgb.early_stopping(800, verbose=True),
                lgb.log_evaluation(500)
            ]
        )

        # Прогнозирование и оценка
        y_pred = self.model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)

        # Сохранение имен признаков
        self.feature_names = X_train.columns.tolist()

        # Визуализация результатов
        self.plot_predictions(y_test, y_pred, "LightGBM_Final")
        self.plot_residuals(y_test, y_pred, "LightGBM")
        self.plot_feature_importance(self.feature_names, top_n=20)

        # Сохранение модели и метрик
        self._save_model_and_metrics(metrics, y_test, y_pred)

        self.logger.info(f"Финальная модель обучена: MAPE: {metrics["MAPE"]:.2f}%")

        return metrics, y_pred

    def _save_model_and_metrics(self, metrics: Dict[str, float], y_test: pd.Series, y_pred: np.ndarray) -> None:
        """Сохранение модели и метрик в файлы"""
        try:
            # Сохранение модели
            model_path = MODELS_PATH / "lgbm_final_model.pkl"
            joblib.dump(self.model, model_path)

            # Сохранение метрик
            metrics_df = pd.DataFrame([metrics])
            metrics_path = REPORTS_PATH / "lgbm_model_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)

            # Сохранение прогнозов для анализа
            predictions_df = pd.DataFrame({
                "actual": y_test,
                "predicted": y_pred
            })
            predictions_path = REPORTS_PATH / "lgbm_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)

            # Сохранение лучших параметров
            params_path = REPORTS_PATH / "lgbm_best_params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)

            self.logger.info("Модель и метрики сохранены:")
            self.logger.info(f"- Модель: {model_path}")
            self.logger.info(f"- Метрики: {metrics_path}")
            self.logger.info(f"- Параметры: {params_path}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели и метрик: {e}")
            raise

    def plot_feature_importance(self, feature_names: List[str], top_n: int = 20) -> None:
        """Визуализация важности признаков"""
        if self.model is None:
            self.logger.warning("Модель не обучена, невозможно построить важность признаков")
            return

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=True).tail(top_n)

        plt.figure(figsize=(12, 10))
        plt.barh(importance["feature"], importance["importance"])
        plt.title(f"Top {top_n} Feature Importance - LightGBM", fontsize=14, fontweight="bold")
        plt.xlabel("Importance", fontsize=12)
        plt.tight_layout()

        # Сохранение графика
        importance_path = REPORTS_PATH / "lgbm_feature_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Сохранение данных важности признаков
        importance_data_path = REPORTS_PATH / "lgbm_feature_importance.csv"
        importance.to_csv(importance_data_path, index=False)

        self.logger.info(f"График важности признаков сохранен: {importance_path}")

    def run_complete_training(self, data_path: Optional[Path] = None, train_time_ratio: float = 0.8) -> Tuple[Union[str, float, None], Optional[np.ndarray]]:
        """Полный цикл обучения модели для использования в пайплайне"""
        try:
            if data_path is None:
                data_path = DATA_PATH / "processed/final_dataset.csv"

            self.logger.info("Загрузка данных для обучения LightGBM модели...")

            # Загрузка данных
            if not data_path.exists():
                raise FileNotFoundError(f"Файл с данными не найден: {data_path}")

            final_data = pd.read_csv(data_path)

            if final_data.empty:
                raise ValueError("Загруженный файл с данными пуст")

            self.logger.info("Разделение данных на обучающую и тестовую выборки...")
            X_train, X_test, y_train, y_test = all_stores_time_split(final_data, train_time_ratio)

            self.logger.info("Запуск полного цикла обучения LightGBM...")
            metrics, predictions = self.train_final_model(X_train, X_test, y_train, y_test)

            self.logger.info("Обучение LightGBM модели успешно завершено")
            return metrics, predictions

        except Exception as e:
            self.logger.error(f"Ошибка выполнения полного цикла обучения: {e}")
            raise

    def load_model(self, model_path: Optional[Path] = None) -> lgb.LGBMRegressor:
        """Загрузка предварительно обученной модели"""
        if model_path is None:
            model_path = DATA_PATH / "models/lgbm_final_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")


        self.model = joblib.load(model_path)
        self.logger.info(f"Модель загружена: {model_path}")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Прогнозирование с использованием обученной модели"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните обучение или загрузку модели.")

        return self.model.predict(X)

def main():
    """Основная функция для запуска обучения LightGBM модели"""
    try:
        logging.basicConfig(level=logging.INFO)
        print("Инциализация обучения LightGBM модели...")

        lgbm_model = LGBMModel()
        metrics, predictions = lgbm_model.run_complete_training()

        print(f"\nОбучение LightGBM модели успешно завершено")
        print(f"Финальные метрики - MAPE: {metrics["MAPE"]:.2f}%, MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}")

        return metrics, predictions

    except Exception as e:
        print(f"Критическая ошибка в обучении LightGBM модели: {e}")
        return None, None

if __name__ == "__main__":
    main()