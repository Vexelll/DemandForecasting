import logging

import joblib
import pandas as pd

from src.database.database_manager import DatabaseManager
from config.settings import REPORTS_PATH, resolve_data_path, setup_logging, get_reporting_config


logger = logging.getLogger(__name__)


def migrate_sales_history(db: DatabaseManager) -> None:
    """Миграция исторических данных из Pickle в SQLite"""
    history_file = resolve_data_path("processed", "sales_history")

    if not history_file.exists():
        logger.warning(f"Файл истории не найден: {history_file}, пропуск миграции")
        return

    try:
        history_df = joblib.load(history_file)

        if not isinstance(history_df, pd.DataFrame) or history_df.empty:
            logger.warning("Файл истории пуст или имеет некорректный формат")
            return

        count = db.save_sales_history(history_df)
        logger.info(f"Мигрировано записей sales_history: {count}")

    except Exception as e:
        logger.error(f"Ошибка миграции sales_history: {e}")

def migrate_predictions(db: DatabaseManager) -> None:
    """Миграция результатов прогнозирования из CSV в SQLite"""
    predictions_file = resolve_data_path("outputs", "predictions")

    if not predictions_file.exists():
        logger.warning(f"Файл прогнозов не найден: {predictions_file}, пропуск миграции")
        return

    try:
        pred_df = pd.read_csv(predictions_file)

        if pred_df.empty:
            logger.warning("Файл прогнозов пуст")
            return

        run_id = "migration_initial"
        count = db.save_predictions(pred_df, run_id)
        logger.info(f"Мигрировано прогнозов: {count} (run_id={run_id})")

    except Exception as e:
        logger.error(f"Ошибка миграции predictions: {e}")

def migrate_model_metrics(db: DatabaseManager) -> None:
    """Миграция метрик LightGBM из CSV в SQLite"""
    report_files = get_reporting_config().get("output_files", {})
    metrics_file = REPORTS_PATH / report_files.get("lgbm_metrics", "lgbm_model_metrics.csv")

    if not metrics_file.exists():
        logger.warning(f"Файл результатов не найден: {metrics_file}, пропуск миграции")
        return

    try:
        metrics_df = pd.read_csv(metrics_file)

        if metrics_df.empty:
            logger.warning("Файл метрик пуст")
            return

        for idx, row in metrics_df.iterrows():
            metrics = {
                "MAPE": row.get("MAPE"),
                "RMSE": row.get("RMSE"),
                "MAE": row.get("MAE"),
                "R2": row.get("R2")
            }

            db.save_model_metrics(
                metrics=metrics,
                model_version=f"migration_lgbm_{idx}",
                n_test_samples=int(row["Samples"]) if pd.notna(row.get("Samples")) else None
            )

        logger.info(f"Мигрировано записей model_metrics: {len(metrics_df)}")

    except Exception as e:
        logger.error(f"Ошибка миграции model_metrics: {e}")

def run_migration() -> None:
    """Запуск полной миграции существующих данных в БД"""
    setup_logging()

    logger.info("=" * 60)
    logger.info("Начало миграции данных в реляционную БД")
    logger.info("=" * 60)

    db = DatabaseManager()

    # Миграция данных
    migrate_sales_history(db)
    migrate_predictions(db)
    migrate_model_metrics(db)

    # Итоговая статистика
    stats = db.get_database_stats()
    logger.info("Миграция завершена. Статистика БД:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    run_migration()
