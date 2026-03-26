import logging
import sqlite3
import uuid
import json
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import DATA_PATH, get_database_config


class DatabaseManager:
    """SQLite хранилище: история продаж, прогнозы, метрики, логи запусков"""
    # Версия схемы - на случай если придется менять таблицы
    DB_VERSION = 1

    def __init__(self, db_path: Path | None = None) -> None:
        self.logger = logging.getLogger(__name__)

        if db_path is None:
            db_filename = get_database_config().get("db_filename", "demand_forecasting.db")
            db_path = DATA_PATH / db_filename

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        """Подключение к SQLite c WAL и настройкой PRAGMA"""
        db_cfg = get_database_config()
        conn = sqlite3.connect(str(self.db_path), timeout=db_cfg.get("connection_timeout", 60))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Создает таблицы, если их еще нет"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # sales_history - замена pickle, теперь можно делать SQL-запросы по датам
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sales_history (
                    store_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    sales REAL NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    promo INTEGER NOT NULL,
                    state_holiday TEXT DEFAULT "0",
                    school_holiday INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (store_id, date)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_store ON sales_history(store_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_history(date)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    run_id TEXT NOT NULL,
                    store_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    predicted_sales REAL NOT NULL,
                    actual_sales REAL,
                    absolute_error REAL,
                    percentage_error REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, store_id, date)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_store_date ON predictions(store_id, date)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    dag_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL,
                    records_processed INTEGER DEFAULT 0,
                    mape REAL, rmse REAL, mae REAL, r2_score REAL,
                    error_message TEXT,
                    model_version TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    mape REAL, rmse REAL, mae REAL, r2_score REAL,
                    n_features INTEGER,
                    n_train_samples INTEGER,
                    n_test_samples INTEGER,
                    best_params TEXT,
                    training_duration_seconds REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("SELECT COUNT(*) FROM schema_version")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (self.DB_VERSION,))

        self.logger.info(f"База данных инициализирована: {self.db_path}")


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(sqlite3.OperationalError),
        before_sleep=lambda retry_state: logging.getLogger(__name__).warning(
            f"БД заблокирована, повторная попытка {retry_state.attempt_number}/3..."
        )
    )
    def save_sales_history(self, df: pd.DataFrame) -> int:
        """Upsert продаж: INSERT OR REPLACE по ключу (store_id, date)"""
        if df is None or len(df) == 0:
            self.logger.warning("Попытка сохранения пустого DataFrame в sales_history")
            return 0

        # DataFrame колонки -> БД колонки
        column_mapping = {
            "Store": "store_id",
            "Date": "date",
            "Sales": "sales",
            "DayOfWeek": "day_of_week",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday"
        }

        available_columns = [col for col in column_mapping if col in df.columns]
        db_df = df[available_columns].copy().rename(columns=column_mapping)

        # SQLite не имеет типа DATE - храним как TEXT в ISO формате (YYYY-MM-DD)
        if "date" in db_df.columns:
            db_df["date"] = pd.to_datetime(db_df["date"]).dt.strftime("%Y-%m-%d")

        saved_count = 0
        with self._get_connection() as conn:
            db_columns = list(db_df.columns)
            records = []
            for row in db_df.itertuples(index=False):
                row_dict = dict(zip(db_columns, row))
                records.append((
                    int(row_dict["store_id"]),
                    row_dict["date"],
                    float(row_dict["sales"]),
                    int(row_dict.get("day_of_week", 0)),
                    int(row_dict.get("promo", 0)),
                    str(row_dict.get("state_holiday", "0")),
                    int(row_dict.get("school_holiday", 0))
                ))

            try:
                conn.executemany("""
                    INSERT OR REPLACE INTO sales_history
                    (store_id, date, sales, day_of_week, promo, state_holiday, school_holiday)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    records
                )
                saved_count = len(records)
            except Exception as e:
                self.logger.error(f"Ошибка записи строки в sales_history: {e}")

        self.logger.info(f"Сохранено в sales_history: {saved_count} записей")
        return saved_count

    def load_sales_history(self) -> pd.DataFrame:
        """Вся история - нужна для расчета лагов"""
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        store_id AS "Store",
                        date AS "Date",
                        sales AS "Sales",
                        day_of_week AS "DayOfWeek",
                        promo AS "Promo",
                        state_holiday AS "StateHoliday",
                        school_holiday AS "SchoolHoliday"
                    FROM sales_history
                    ORDER BY store_id, date""",
                    conn,
                    parse_dates=["Date"]
                )
                return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки sales_history: {e}")
            return pd.DataFrame()

    def get_store_history(self, store_id: int, days_back: int = 365, end_date: str | None = None) -> pd.DataFrame:
        """История одного магазина за N дней назад от end_date"""
        try:
            if end_date is None:
                end_date_clause = "(SELECT MAX(date) FROM sales_history WHERE store_id = ?)"
                params = [store_id, store_id, days_back, store_id]
            else:
                end_date_clause = "?"
                params = [store_id, end_date, days_back, end_date]

            query = f"""
                SELECT
                    store_id AS "Store",
                    date AS "Date",
                    sales AS "Sales",
                    day_of_week AS "DayOfWeek",
                    promo AS "Promo",
                    state_holiday AS "StateHoliday",
                    school_holiday AS "SchoolHoliday"
                FROM sales_history
                WHERE store_id = ?
                    AND date >= date({end_date_clause}, "-" || ? || " days")
                    AND date <= {end_date_clause}
                ORDER BY date
            """

            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=["Date"])
                return df

        except Exception as e:
            self.logger.error(f"Ошибка получения истории магазина {store_id}: {e}")
            return pd.DataFrame()

    def get_sales_history_stats(self) -> dict[str, Any]:
        """Кол-во записей, магазинов, диапазон дат, размер файла"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM sales_history")
                total_records = cursor.fetchone()[0]

                if total_records == 0:
                    return {
                        "total_records": 0,
                        "unique_stores": 0,
                        "date_range": "База данных пуста",
                        "data_size_mb": 0,
                        "date_coverage_days": 0
                    }

                cursor.execute("SELECT COUNT(DISTINCT store_id) FROM sales_history")
                unique_stores = cursor.fetchone()[0]

                cursor.execute("SELECT MIN(date), MAX(date) FROM sales_history")
                min_date, max_date = cursor.fetchone()

                file_size = self.db_path.stat().st_size / (1024 * 1024)

                return {
                    "total_records": total_records,
                    "unique_stores": unique_stores,
                    "date_range": f"{min_date} - {max_date}",
                    "data_size_mb": round(file_size, 2),
                    "date_coverage_days": (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days
                }
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики sales_history: {e}")
            return {"total_records": 0, "error": str(e)}

    def delete_old_sales_history(self, cutoff_date: str) -> int:
        """Чистка старых данных по cutoff_date"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM sales_history WHERE date < ?",
                    (cutoff_date,)
                )
                deleted = cursor.rowcount
                self.logger.info(f"Удалено из sales_history: {deleted} записей (cutoff={cutoff_date})")
                return deleted
        except Exception as e:
            self.logger.error(f"Ошибка удаления устаревших записей: {e}")
            return 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(sqlite3.OperationalError)
    )
    def save_predictions(self, df: pd.DataFrame, run_id: str) -> int:
        """Прогнозы в таблицу predictions, привязаны к run_id"""
        if df is None or len(df) == 0:
            self.logger.warning("Попытка сохранения пустого DataFrame в predictions")
            return 0

        column_mapping = {
            "Store": "store_id",
            "Date": "date",
            "PredictedSales": "predicted_sales",
            "ActualSales": "actual_sales",
            "AbsoluteError": "absolute_error",
            "PercentageError": "percentage_error"
        }

        db_df = df.copy()
        db_df = db_df.rename(columns={k: v for k, v in column_mapping.items() if k in db_df.columns})

        if "date" in db_df.columns:
            db_df["date"] = pd.to_datetime(db_df["date"]).dt.strftime("%Y-%m-%d")

        saved_count = 0
        with self._get_connection() as conn:
            db_columns = list(db_df.columns)
            records = []
            for row in db_df.itertuples(index=False):
                row_dict = dict(zip(db_columns, row))
                records.append((
                    run_id,
                    int(row_dict.get("store_id", 0)),
                    row_dict.get("date", ""),
                    float(row_dict.get("predicted_sales", 0)),
                    float(row_dict.get("actual_sales", 0)) if pd.notna(row_dict.get("actual_sales")) else None,
                    float(row_dict.get("absolute_error", 0)) if pd.notna(row_dict.get("absolute_error")) else None,
                    float(row_dict.get("percentage_error", 0)) if pd.notna(row_dict.get("percentage_error")) else None,
                ))

            try:
                conn.executemany(
                    """INSERT INTO predictions
                    (run_id, store_id, date, predicted_sales, actual_sales, absolute_error, percentage_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    records
                )
                saved_count = len(records)
            except Exception as e:
                self.logger.error(f"Ошибка записи строки в predictions: {e}")

        self.logger.info(f"Сохранено в predictions: {saved_count} записей (run_id={run_id})")
        return saved_count

    def get_predictions_by_run(self, run_id: str) -> pd.DataFrame:
        """Прогнозы конкретного run_id с маппингом колонок обратно в CamelCase"""
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                            store_id AS "Store",
                            date AS "Date",
                            predicted_sales AS "PredictedSales",
                            actual_sales AS "ActualSales",
                            absolute_error AS "AbsoluteError",
                            percentage_error AS "PercentageError"
                        FROM predictions
                        WHERE run_id = ?
                        ORDER BY store_id, date""",
                        conn,
                        params=[run_id],
                        parse_dates=["Date"]
                )
                return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки прогнозов для run_id={run_id}: {e}")
            return pd.DataFrame()

    def get_latest_predictions(self) -> pd.DataFrame:
        """Последний run -> прогнозы"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT run_id FROM predictions ORDER BY rowid DESC LIMIT 1")
                row = cursor.fetchone()

                if row is None:
                    return pd.DataFrame()

                return self.get_predictions_by_run(row[0])
        except Exception as e:
            self.logger.error(f"Ошибка загрузки последних прогнозов: {e}")
            return pd.DataFrame()

    def list_prediction_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Список run_id с мета: кол-во записей, даты, avg mape"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        run_id,
                        COUNT(*) as records,
                        MIN(date) as date_from,
                        MAX(date) as date_to,
                        AVG(percentage_error) as avg_mape,
                        MIN(created_at) as created_at
                    FROM predictions
                    GROUP BY run_id
                    ORDER BY MAX(rowid) DESC
                    LIMIT ?""",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Ошибка получения списка запусков: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(sqlite3.OperationalError)
    )
    def log_pipeline_run(self, run_id: str, dag_name: str, status: str, started_at: str | None = None, finished_at: str | None = None, duration_seconds: float | None = None, records_processed: int = 0,
        mape: float | None = None, rmse: float | None = None, mae: float | None = None, r2_score: float | None = None, error_message: str | None = None, model_version: str | None = None) -> None:
        """Пишет в pipeline_runs: run_id, статус, метрики, длительность"""
        if started_at is None:
            started_at = datetime.now().isoformat()
        if finished_at is None:
            finished_at = datetime.now().isoformat()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO pipeline_runs
                    (run_id, dag_name, status, started_at, finished_at, duration_seconds, records_processed,
                    mape, rmse, mae, r2_score, error_message, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, dag_name, status, started_at, finished_at, duration_seconds, records_processed,
                        mape, rmse, mae, r2_score, error_message, model_version
                    )
                )
            self.logger.info(f"Запуск пайплайна зафиксирован: run_id={run_id}, dag={dag_name}, status={status}")
        except Exception as e:
            self.logger.error(f"Ошибка логирования запуска пайплайна: {e}")

    def get_pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние N запусков из pipeline_runs"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?""",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Ошибка получения истории запусков: {e}")
            return []

    def save_model_metrics(self, metrics: dict[str, Any], model_version: str | None = None, n_features: int | None = None, n_train_samples: int | None = None, n_test_samples: int | None = None,
        best_params: dict | None = None, training_duration_seconds: float | None = None) -> None:
        """INSERT в model_metrics после каждого обучения"""
        if model_version is None:
            model_version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        params_json = json.dumps(best_params, ensure_ascii=False) if best_params else None

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO model_metrics
                    (trained_at, model_version, mape, rmse, mae, r2_score, n_features, n_train_samples, n_test_samples, best_params, training_duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(), model_version, metrics.get("MAPE"), metrics.get("RMSE"), metrics.get("MAE"), metrics.get("R2"),
                        n_features, n_train_samples, n_test_samples, params_json, training_duration_seconds,
                    )
                )
            self.logger.info(f"Метрики модели сохранены: version={model_version}, MAPE={metrics.get("MAPE", "N/A")}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения метрик модели: {e}")

    def get_model_metrics_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Тренд MAPE/RMSE по обучениям - для дашборда"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                """SELECT * FROM model_metrics ORDER BY trained_at DESC LIMIT ?""",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Ошибка получения истории метрик: {e}")
            return []

    @staticmethod
    def generate_run_id() -> str:
        """run_YYYYMMDD_<6hex> - уникальный ID запуска"""
        return f"run_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}"

    def get_database_stats(self) -> dict[str, Any]:
        """Размеры таблиц + вес файла БД"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                stats = {}

                for table in ["sales_history", "predictions", "pipeline_runs", "model_metrics"]:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                stats["db_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
                stats["db_path"] = str(self.db_path)

                return stats
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики БД: {e}")
            return {"error": str(e)}

    @staticmethod
    def create_database_manager(logger=None):
        """Пробует создать DatabaseManager, при ошибке -> None"""
        if logger is None:
            logger = logging.getLogger(__name__)
        try:
            db = DatabaseManager()
            logger.debug("DatabaseManager инициализирован")
            return db
        except Exception as e:
            logger.warning(f"Ошибка инициализации БД, fallback на файлы: {e}")
            return None
