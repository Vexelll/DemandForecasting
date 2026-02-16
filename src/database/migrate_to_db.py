import logging
import json
import joblib
import pandas as pd
from config.settings import DATA_PATH, REPORTS_PATH
from src.database.database_manager import DatabaseManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def migrate_sales_history(db: DatabaseManager) -> None:
    """Миграция исторических данных из Pickle в SQLite"""
    history_file = DATA_PATH / "processed/sales_history.pkl"

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
    predictions_file = DATA_PATH / "outputs/predictions.csv"

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
    """Миграция метрик модели из JSON в SQLite"""
    results_file = REPORTS_PATH / "model_results.json"

    if not results_file.exists():
        logger.warning(f"Файл результатов не найден: {results_file}, пропуск миграции")
        return

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Сохраняем метрики лучшей модели
        metrics = {
            "MAPE": results.get("best_mape"),
            "R2": results.get("best_r2"),
            "MAE": None,
            "RMSE": None
        }

        db.save_model_metrics(
            metrics=metrics,
            model_version=f"migration_{results.get("best_model", "unknown")}"
        )
        logger.info("Мигрированы метрики модели из model_results.json")

    except Exception as e:
        logger.error(f"Ошибка миграции model_metrics: {e}")

def run_migration() -> None:
    """Запуск полной миграции существующих данных в БД"""
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
