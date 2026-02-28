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

from config.settings import DATA_PATH, MODELS_PATH, REPORTS_PATH, all_stores_time_split, get_model_config, get_optimization_config, get_reporting_config, setup_logging
from src.models.base_model import BaseModels


class LGBMModel(BaseModels):
    def __init__(self):
        super().__init__()
        self.model: Optional[lgb.LGBMRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None
        self.feature_names: List[str] = []
        self.logger = logging.getLogger(__name__)

        model_cfg = get_model_config()
        self.best_n_estimators: int = model_cfg.get("n_estimators_max", 5000)

    @staticmethod
    def _build_time_key(X: pd.DataFrame) -> pd.Series:
        """Year+Week -> datetime по ISO, для корректного TimeSeriesSplit"""
        time_key = pd.to_datetime(
            X["Year"].astype(str) + "-W" + X["Week"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u"
        )
        return time_key

    @staticmethod
    def _detect_device() -> Dict[str, Any]:
        """Пробует GPU, fallback на CPU"""
        try:
            test_model = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbosity=-1)
            test_model.fit(
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2])
            )
            return {
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0
            }
        except Exception:
            return {"device": "cpu"}

    def _validate_input_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        """Размеры X/y должны совпадать, данных не должно быть 0"""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Все входные данные должны быть предоставлены")

        if len(X_train) != len(y_train):
            raise ValueError(f"Несоответствие размеров X_train ({len(X_train)}) и y_train ({len(y_train)})")

        if len(X_test) != len(y_test):
            raise ValueError(f"Несоответствие размеров X_test ({len(X_test)}) и y_test ({len(y_test)})")

        if len(X_train) < 100:
            self.logger.warning("Малый объем обучающих данных может повлиять на качество модели")

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective: cv по Year-Week с early stopping"""

        model_cfg = get_model_config()
        ss = model_cfg.get("search_space", {})
        cv_cfg = model_cfg.get("cv", {})

        device_params = self._detect_device()

        params = {
            "objective": "regression",
            "metric": "mape",
            "verbosity": -1,
            "boosting_type": "gbdt",
            **device_params,
            "lambda_l1": trial.suggest_float("lambda_l1", *ss.get("lambda_l1", [0.1, 0.50]), log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", *ss.get("lambda_l2", [0.1, 0.50]), log=True),
            "num_leaves": trial.suggest_int("num_leaves", *ss.get("num_leaves", [31, 255])),
            "max_depth": trial.suggest_int("max_depth", *ss.get("max_depth", [4, 15])),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *ss.get("min_data_in_leaf", [50, 500])),
            "feature_fraction": trial.suggest_float("feature_fraction", *ss.get("feature_fraction", [0.4, 0.95])),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *ss.get("bagging_fraction", [0.4, 0.95])),
            "bagging_freq": trial.suggest_int("bagging_freq", *ss.get("bagging_freq", [1, 7])),
            "min_child_weight": trial.suggest_float("min_child_weight", *ss.get("min_child_weight", [1e-5, 10.0]), log=True),
            "learning_rate": trial.suggest_float("learning_rate", *ss.get("learning_rate", [0.005, 0.3]), log=True),
            "max_bin": trial.suggest_int("max_bin", *ss.get("max_bin", [200, 255])),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "path_smooth": trial.suggest_float("path_smooth", *ss.get("path_smooth", [1.0, 50.0])),
            "random_state": model_cfg.get("random_state", 42)
        }

        # Группировка по Year-Week - чтобы TimeSeriesSplit не резал недели
        time_key = self._build_time_key(X)
        unique_periods = sorted(time_key.unique())

        period_indices = np.arange(len(unique_periods))

        n_estimators_max = model_cfg.get("n_estimators_max", 5000)
        early_stopping_rounds = model_cfg.get("early_stopping_rounds", 500)

        tscv = TimeSeriesSplit(cv_cfg.get("n_splits", 3))
        scores = []
        best_iterations = []

        for fold_idx, (train_period_idx, val_period_idx) in enumerate(tscv.split(period_indices)):
            train_periods = [unique_periods[i] for i in train_period_idx]
            val_periods = [unique_periods[i] for i in val_period_idx]

            train_mask = time_key.isin(train_periods)
            val_mask = time_key.isin(val_periods)

            X_fold_train, y_fold_train = X[train_mask], y[train_mask]
            X_fold_val, y_fold_val = X[val_mask], y[val_mask]

            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators_max)
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric="mape",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(False)
                ],
            )

            y_pred = model.predict(X_fold_val)
            mape = np.mean(np.abs((y_fold_val - y_pred) / np.maximum(y_fold_val, 1))) * 100
            scores.append(mape)
            best_iterations.append(model.best_iteration_)

            trial.report(mape, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Медиана best_iteration_ по фолдам - компромисс между переобучением и недообучением
            trial.set_user_attr("best_n_estimators", int(np.median(best_iterations)))

        return np.mean(scores)

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: Optional[int] = None) -> Dict[str, Any]:
        """TPE sampler + MedianPruner - отсекает слабые trials рано"""
        self.logger.info("Начало оптимизации гиперпараметров LightGBM")

        model_cfg = get_model_config()
        opt_cfg = get_optimization_config()
        pruner_cfg = opt_cfg.get("pruner", {})

        if n_trials is None:
            n_trials = opt_cfg.get("n_trials", 50)

        sampler = optuna.samplers.TPESampler(seed=opt_cfg.get("sampler_seed", 42))

        # MedianPruner: если trial хуже медианы завершенных - отсекаем
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 10),  # Минимум trials перед началом pruning
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 1),  # Пропустить первые 2 фолда перед pruning
            interval_steps=pruner_cfg.get("interval_steps", 1),  # Проверка pruning на каждом фолде
        )

        self.study = optuna.create_study(direction=opt_cfg.get("direction", "minimize"), sampler=sampler, pruner=pruner)

        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            show_progress_bar=True,  # Прогресс-бар
        )

        n_estimators_max = model_cfg.get("n_estimators_max", 5000)
        self.best_params = self.study.best_params
        self.best_n_estimators = self.study.best_trial.user_attrs.get("best_n_estimators", n_estimators_max)

        device_params = self._detect_device()

        self.best_params.update(
            {
                "objective": "regression",
                "metric": "mape",
                "verbosity": -1,
                "boosting_type": "gbdt",
                **device_params,
                "n_estimators": self.best_n_estimators,
                "random_state": model_cfg.get("random_state", 42),
            }
        )

        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        complete_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        self.logger.info(f"Оптимизация завершена. Лучший MAPE: {self.study.best_value:.2f}%")
        self.logger.info(f"Лучшие параметры: {self.best_params}")
        self.logger.info(f"Лучшее число деревьев (медиана CV): {self.best_n_estimators}")
        self.logger.info(f"Trials: {complete_trials} завершено, {pruned_trials} отсечено (pruned)")

        return self.best_params

    def train_final_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
        """Train на лучших Optuna-параметрах + визуализация + сохранение"""

        self._validate_input_data(X_train, X_test, y_train, y_test)

        self.logger.info("Начало обучения финальной модели LightGBM")

        model_cfg = get_model_config()
        report_cfg = get_reporting_config()
        early_stopping_rounds = model_cfg.get("early_stopping_rounds", 500)
        top_n = report_cfg.get("feature_importance_top_n", 20)

        if self.best_params is None:
            self.best_params = self.optimize_hyperparameters(X_train, y_train)

        self.logger.info(f"Train: {len(X_train)} строк, Test (eval): {len(X_test)} строк")

        self.model = lgb.LGBMRegressor(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mape",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=True),
                lgb.log_evaluation(200)
            ]
        )

        y_pred = self.model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)

        self.feature_names = X_train.columns.tolist()

        self.plot_predictions(y_test, y_pred, "LightGBM_Final")
        self.plot_residuals(y_test, y_pred, "LightGBM")
        self.plot_feature_importance(self.feature_names, top_n=top_n)

        self._save_model_and_metrics(metrics, y_test, y_pred)

        self.logger.info(f"Финальная модель обучена: MAPE: {metrics["MAPE"]:.2f}%")

        return metrics, y_pred

    def _save_model_and_metrics(self, metrics: Dict[str, float], y_test: pd.Series, y_pred: np.ndarray) -> None:
        """model.pkl + metrics.csv + params.json"""
        try:
            model_cfg = get_model_config()
            report_files = get_reporting_config().get("output_files", {})

            model_path = MODELS_PATH / model_cfg.get("model_filename", "lgbm_final_model.pkl")
            joblib.dump(self.model, model_path)

            metrics_df = pd.DataFrame([metrics])
            metrics_path = REPORTS_PATH / report_files.get("lgbm_metrics", "lgbm_model_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)

            predictions_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
            predictions_path = REPORTS_PATH / report_files.get("lgbm_predictions", "lgbm_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)

            params_path = REPORTS_PATH / report_files.get("lgbm_best_params", "lgbm_best_params.json")

            serializable_params = {}
            for key, value in self.best_params.items():
                if isinstance(value, (np.integer,)):
                    serializable_params[key] = int(value)
                elif isinstance(value, (np.floating,)):
                    serializable_params[key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_params[key] = bool(value)
                else:
                    serializable_params[key] = value

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(serializable_params, f, indent=2, ensure_ascii=False)

            self.logger.info("Модель и метрики сохранены:")
            self.logger.info(f"- Модель: {model_path}")
            self.logger.info(f"- Метрики: {metrics_path}")
            self.logger.info(f"- Параметры: {params_path}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели и метрик: {e}")
            raise

    def run_complete_training(self, data_path: Optional[Path] = None, train_time_ratio: Optional[float] = None) -> Tuple[Dict[str, float], np.ndarray]:
        """Загрузка csv -> optimize -> train -> save, все за один вызов"""
        try:
            if data_path is None:
                data_path = DATA_PATH / "processed/final_dataset.csv"

            if train_time_ratio is None:
                train_time_ratio = get_model_config().get("train_time_ratio", 0.8)

            self.logger.info("Загрузка данных...")

            if not data_path.exists():
                raise FileNotFoundError(f"Файл с данными не найден: {data_path}")

            final_data = pd.read_csv(data_path)

            if final_data.empty:
                raise ValueError("Загруженный файл с данными пуст")

            self.logger.info("Split на train/test...")
            X_train, X_test, y_train, y_test = all_stores_time_split(final_data, train_time_ratio)

            self.logger.info("Обучение LightGBM...")
            metrics, predictions = self.train_final_model(X_train, X_test, y_train, y_test)

            self.logger.info("Обучение завершено")
            return metrics, predictions

        except Exception as e:
            self.logger.error(f"Ошибка выполнения полного цикла обучения: {e}")
            raise

    def load_model(self, model_path: Optional[Path] = None) -> lgb.LGBMRegressor:
        """joblib.load из models/"""
        if model_path is None:
            model_filename = get_model_config().get("model_filename", "lgbm_final_model.pkl")
            model_path = MODELS_PATH / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self.model = joblib.load(model_path)
        self.logger.info(f"Модель загружена: {model_path}")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """model.predict(X) - модель должна быть обучена или загружена"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните обучение или загрузку модели")

        return self.model.predict(X)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        logger.info("Запуск обучения LightGBM...")

        lgbm_model = LGBMModel()
        metrics, predictions = lgbm_model.run_complete_training()

        logger.info(f"Финальные метрики - MAPE: {metrics["MAPE"]:.2f}%, MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}")

        return metrics, predictions

    except Exception as e:
        logger.error(f"Критическая ошибка в обучении LightGBM модели: {e}")
        return None, None


if __name__ == "__main__":
    main()
