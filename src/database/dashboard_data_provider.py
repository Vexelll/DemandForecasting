import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.database.database_manager import DatabaseManager


class DashboardDataProvider:
    """Читает прогнозы из БД для дашборда, fallback на демо-данные"""

    REQUIRED_PREDICTION_COLUMNS = ["Store", "Date", "PredictedSales", "ActualSales"]

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.db = DatabaseManager.create_database_manager()

        if self.db is not None:
            self.data_source = "database"
            self.logger.info("DashboardDataProvider: источник данных - SQLite БД")
        else:
            self.data_source = "unavailable"
            self.logger.warning("DashboardDataProvider: БД недоступна, данные из CSV не загружаются. Используются демо-данные")

    def load_predictions(self, run_id: str | None = None) -> pd.DataFrame:
        """Последний run или конкретный run_id -> DataFrame для графиков"""
        if self.db is None:
            self.logger.warning("БД недоступна, возвращаются демо-данные")
            return self._create_demo_data()

        try:
            if run_id is not None:
                data = self.db.get_predictions_by_run(run_id)
            else:
                data = self.db.get_latest_predictions()

            if data is None or len(data) == 0:
                self.logger.warning("В БД нет прогнозов, возвращаются демо-данные")
                return self._create_demo_data()

            if not self._validate_predictions(data):
                self.logger.warning("Данные из БД не прошли валидацию, демо-данные")
                return self._create_demo_data()

            data = self._enrich_predictions(data)

            self.logger.info(f"Прогнозы загружены из БД: {len(data)} записей, {data["Store"].nunique()} магазинов")
            return data

        except Exception as e:
            self.logger.error(f"Ошибка загрузки прогнозов из БД: {e}")
            return self._create_demo_data()

    def load_predictions_for_comparison(self, run_id_1: str, run_id_2: str) -> pd.DataFrame:
        """Два run_id -> merge по Store+Date -> дельта прогнозов"""
        if self.db is None:
            self.logger.warning("БД недоступна, сравнение невозможно")
            return pd.DataFrame()

        try:
            pred_1 = self.db.get_predictions_by_run(run_id_1)
            pred_2 = self.db.get_predictions_by_run(run_id_2)

            if pred_1.empty or pred_2.empty:
                self.logger.warning("Один из запусков не содержит данных")
                return pd.DataFrame()

            comparison = pd.merge(
                pred_1[["Store", "Date", "PredictedSales", "ActualSales"]],
                pred_2[["Store", "Date", "PredictedSales"]],
                on=["Store", "Date"],
                how="inner",
                suffixes=("_v1", "_v2")
            )

            comparison["PredictionDelta"] = (comparison["PredictedSales_v2"] - comparison["PredictedSales_v1"])
            comparison["DeltaPct"] = np.where(comparison["PredictedSales_v1"] > 0, (comparison["PredictionDelta"] / comparison["PredictedSales_v1"]) * 100, 0.0)

            self.logger.info(f"Сравнение загружено: {len(comparison)} записей, run_id_1={run_id_1}, run_id_2={run_id_2}")
            return comparison

        except Exception as e:
            self.logger.error(f"Ошибка загрузки сравнения: {e}")
            return pd.DataFrame()

    def get_available_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Список run_id с мета: записи, даты, avg mape"""
        if self.db is None:
            return []

        try:
            return self.db.list_prediction_runs(limit=limit)
        except Exception as e:
            self.logger.error(f"Ошибка получения списка запусков: {e}")
            return []

    def get_pipeline_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """pipeline_runs для вкладки мониторинга"""
        if self.db is None:
            return []

        try:
            return self.db.get_pipeline_runs(limit=limit)
        except Exception as e:
            self.logger.error(f"Ошибка получения истории пайплайна: {e}")
            return []

    def get_model_metrics_trend(self, limit: int = 20) -> list[dict[str, Any]]:
        """mape/rmse по обучениям для графиков тренда"""
        if self.db is None:
            return []

        try:
            return self.db.get_model_metrics_history(limit=limit)
        except Exception as e:
            self.logger.error(f"Ошибка получения истории метрик: {e}")
            return []

    def get_data_source_info(self) -> dict[str, Any]:
        """Для футера: источник, размер БД, кол-во записей"""
        info = {
            "source": self.data_source,
            "timestamp": datetime.now().isoformat()
        }

        if self.db is not None:
            try:
                stats = self.db.get_database_stats()
                info.update({
                    "db_path": stats.get("db_path", "N/A"),
                    "db_size_mb": stats.get("db_size_mb", 0),
                    "predictions_count": stats.get("predictions_count", 0),
                    "pipeline_runs_count": stats.get("pipeline_runs_count", 0),
                    "model_metrics_count": stats.get("model_metrics_count", 0)
                })
            except Exception as e:
                self.logger.warning(f"Ошибка получения статистики БД: {e}")

        return info

    def _validate_predictions(self, data: pd.DataFrame) -> bool:
        """Проверяет, что Store, Date, PredictedSales, ActualSales есть"""
        missing = [col for col in self.REQUIRED_PREDICTION_COLUMNS if col not in data.columns]

        if missing:
            self.logger.warning(f"Отсутствуют обязательные колонки: {missing}")
            return False

        if len(data) == 0:
            return False

        return True

    def _enrich_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Досчитывает AbsoluteError и PercentageError, если их нет"""
        df = data.copy()

        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])

        # AbsoluteError
        if "AbsoluteError" not in df.columns:
            if "ActualSales" in df.columns and "PredictedSales" in df.columns:
                df["AbsoluteError"] = np.abs(df["ActualSales"] - df["PredictedSales"])

        # PercentageError
        if "PercentageError" not in df.columns:
            if "AbsoluteError" in df.columns and "ActualSales" in df.columns:
                df["PercentageError"] = np.where(df["ActualSales"] > 0, (df["AbsoluteError"] / df["ActualSales"]) * 100, 0.0)

        return df

    def _create_demo_data(self) -> pd.DataFrame:
        """Синтетика для тестирования дашборда, когда БД пуста"""
        self.logger.info("Генерация демо-данных для дашборда")

        rng = np.random.default_rng(seed=42)

        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")
        num_stores = 2
        all_rows = []

        for store_id in range(1, num_stores + 1):
            base_sales = 8000 + store_id * 2000
            seasonal_effect = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            trend = 50 * np.arange(len(dates))
            noise = rng.normal(0, 500, len(dates))

            actual_sales = base_sales + seasonal_effect + trend + noise
            predicted_sales = actual_sales + rng.normal(0, 800, len(dates))

            store_df = pd.DataFrame({
                "Store": store_id,
                "Date": dates,
                "PredictedSales": np.maximum(predicted_sales, 0),
                "ActualSales": np.maximum(actual_sales, 0)
            })
            all_rows.append(store_df)

        demo_data = pd.concat(all_rows, ignore_index=True)

        # Дополнение рассчитываемых колонок
        demo_data = self._enrich_predictions(demo_data)

        self.logger.info(f"Демо-данные сгенерированы: {len(demo_data)} записей, {num_stores} магазинов")

        return demo_data
