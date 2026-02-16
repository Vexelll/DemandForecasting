import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import DATA_PATH, MODELS_PATH, REPORTS_PATH, all_stores_time_split
from src.models.base_model import BaseModels


class LGBMModel(BaseModels):
    def __init__(self):
        super().__init__()
        self.model: Optional[lgb.LGBMRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None
        self.feature_names: List[str] = []
        self.logger = logging.getLogger(__name__)

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
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_bin": trial.suggest_int("max_bin", 255, 500),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
            "random_state": 42,
        }

        # Создаём временной ключ и группируем данные
        time_key = pd.to_datetime(X["Year"].astype(str) + "-W" + X["Week"].astype(str).str.zfill(2) + "-1", format="%Y-W%W-%w")
        unique_periods = sorted(time_key.unique())

        # Создаём массив индексов периодов (каждый период = один индекс)
        period_indices = np.arange(len(unique_periods))

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold_idx, (train_period_idx, val_period_idx) in enumerate(tscv.split(period_indices)):
            # Получаем периоды для train и validation
            train_periods = [unique_periods[i] for i in train_period_idx]
            val_periods = [unique_periods[i] for i in val_period_idx]

            # Маски для фильтрации данных по периодам
            train_mask = time_key.isin(train_periods)
            val_mask = time_key.isin(val_periods)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model = lgb.LGBMRegressor(**params, n_estimators=2000)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mape",
                callbacks=[
                    lgb.early_stopping(300, verbose=False),
                    lgb.log_evaluation(False)
                ],
            )

            y_pred = model.predict(X_val)
            mape = np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100
            scores.append(mape)

            trial.report(mape, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Оптимизация гиперпараметров с использованием Optuna"""
        self.logger.info("Начало оптимизации гиперпараметров LightGBM")

        sampler = optuna.samplers.TPESampler(seed=42)

        # Добавляем MedianPruner для раннего отсечения неперспективных trials
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Минимум trials перед началом pruning
            n_warmup_steps=2,  # Пропустить первые 2 фолда перед pruning
            interval_steps=1,  # Проверка pruning на каждом фолде
        )

        self.study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            show_progress_bar=True,  # Прогресс-бар
        )

        self.best_params = self.study.best_params
        self.best_params.update(
            {
                "objective": "regression",
                "metric": "mape",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": 20000,
                "random_state": 42,
            }
        )

        # Логирование статистики pruning
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        complete_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        self.logger.info(f"Оптимизация завершена. Лучший MAPE: {self.study.best_value:.2f}%")
        self.logger.info(f"Лучшие параметры: {self.best_params}")
        self.logger.info(f"Trials: {complete_trials} завершено, {pruned_trials} отсечено (pruned)")

        return self.best_params

    def train_final_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
        """Обучение финальной модели на лучших параметрах"""

        # Валидация входных данных
        self._validate_input_data(X_train, X_test, y_train, y_test)

        self.logger.info("Начало обучения финальной модели LightGBM")

        # Оптимизация гиперпараметров если не выполнена
        if self.best_params is None:
            self.best_params = self.optimize_hyperparameters(X_train, y_train)

        # Создаём временной ключ
        time_key = pd.to_datetime(X_train["Year"].astype(str) + "-W" + X_train["Week"].astype(str).str.zfill(2) + "-1", format="%Y-W%W-%w")
        unique_periods = sorted(time_key.unique())

        # Берём последние 15% периодов для валидации
        val_period_count = max(1, int(len(unique_periods) * 0.15))
        train_periods = unique_periods[:-val_period_count]
        val_periods = unique_periods[-val_period_count:]

        # Фильтруем данные по периодам
        train_mask = time_key.isin(train_periods)
        val_mask = time_key.isin(val_periods)

        X_train_actual, y_train_actual = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        self.logger.info(f"Train: {len(train_periods)} периодов ({len(X_train_actual)} строк)")
        self.logger.info(f"Val: {len(val_periods)} периодов ({len(X_val)} строк)")

        # Обучение финальной модели
        self.model = lgb.LGBMRegressor(**self.best_params)
        self.model.fit(
            X_train_actual, y_train_actual,
            eval_set=[(X_val, y_val)],
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
            predictions_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
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

    def run_complete_training(self, data_path: Optional[Path] = None, train_time_ratio: float = 0.8) -> Tuple[Dict[str, float], np.ndarray]:
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
            model_path = MODELS_PATH / "lgbm_final_model.pkl"

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"

    )
    logger = logging.getLogger(__name__)
    try:
        logger.info("Инициализация обучения LightGBM модели...")

        lgbm_model = LGBMModel()
        metrics, predictions = lgbm_model.run_complete_training()

        logger.info(f"Финальные метрики - MAPE: {metrics['MAPE']:.2f}%, MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

        return metrics, predictions

    except Exception as e:
        logger.error(f"Критическая ошибка в обучении LightGBM модели: {e}")
        return None, None


if __name__ == "__main__":
    main()
