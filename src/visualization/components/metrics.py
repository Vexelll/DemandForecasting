import logging
from typing import Dict, Any, List

from dash import html
import numpy as np

logger = logging.getLogger(__name__)

def calculate_metrics(data) -> Dict[str, Any]:
    """Расчет метрик качества прогнозирования с проверкой данных"""
    # Базовые метрики по умолчанию
    default_metrics = {
        "mape": 0.0, "mae": 0.0, "rmse": 0.0,
        "total_sales": 0.0, "bias": 0.0, "std_error": 0.0,
        "data_points": 0.0
    }

    # Проверка наличия данных
    if data is None or len(data) == 0:
        logger.debug("Расчет метрик: данные пусты, возвращены значения по умолчанию")
        return default_metrics

    # Проверка наличия колонок
    required_columns = ["ActualSales", "PredictedSales"]
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        logger.warning(f"Расчет метрик: отсутствуют колонки: {missing}")
        return default_metrics

    # Извлечение данных и удаление NaN
    valid_mask = data["ActualSales"].notna() & data["PredictedSales"].notna()
    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        logger.debug("Расчет метрик: нет валидных данных после фильтрации NaN")
        return default_metrics

    actual = valid_data["ActualSales"].values
    predicted = valid_data["PredictedSales"].values

    # Расчет ошибки
    errors = actual - predicted

    # Безопасный расчет MAPE (иключаем нулевые продажи)
    with np.errstate(divide="ignore", invalid ="ignore"):
        non_zero_mask = actual > 0
        if non_zero_mask.sum() > 0:
            ape = np.abs(errors[non_zero_mask]) / actual[non_zero_mask] * 100
            mape = np.mean(ape[np.isfinite(ape)])
        else:
            mape = 0.0

    # Расчет метрик
    metrics = {
        "mape": round(mape, 2) if np.isfinite(mape) else 0.0,
        "mae": round(np.mean(np.abs(errors)), 2),
        "rmse": round(np.sqrt(np.mean(errors ** 2)), 2),
        "total_sales": round(actual.sum(), 2),
        "bias": round(errors.mean(), 2),
        "std_error": round(errors.std(), 2) if len(errors) > 1 else 0.0,
        "data_points": len(valid_data)
    }

    logger.debug(f"Метрики рассчитаны: MAPE={metrics["mape"]}%, MAE={metrics["mae"]}, RMSE={metrics["rmse"]}, точек данных={metrics["data_points"]}")

    return metrics


def create_metric_cards(metrics: Dict[str, Any]) -> List[html.Div]:
    """Создание карточек с метриками с цветовой индикацией качества"""

    # Определение цвета для MAPE в зависимости от качества
    if metrics["mape"] < 10:
        mape_color = "#27ae60"  # отличное качество
    elif metrics["mape"] < 15:
        mape_color = "#f39c12"  # хорошее качество
    else:
        mape_color = "#e74c3c"  # требует улучшения

    # Цвета для других метрик
    mae_color = "#3498db"
    rmse_color = "#9b59b6"
    sales_color = "#2ecc71"

    mape_card = _create_metric_card(
        "Точность прогноза (MAPE)",
        f"{metrics["mape"]:.2f}%",
        mape_color,
        f"На основе {metrics["data_points"]} точек данных"
    )

    mae_card = _create_metric_card(
        "Средняя абсолютная ошибка",
        f"{metrics["mae"]:,.0f} €",
        mae_color,
        f"Смещение: {metrics["bias"]:+,.0f} €"
    )

    rmse_card = _create_metric_card(
        "Среднеквадратичная ошибка",
        f"{metrics["rmse"]:,.0f} €",
        rmse_color,
        f"Стд. откл. ошибки: {metrics["std_error"]:,.0f} €"
    )

    avg_daily = metrics["total_sales"] / max(metrics["data_points"], 1)
    sales_card = _create_metric_card(
        "Общий объем продаж",
        f"{metrics["total_sales"]:,.0f} €",
        sales_color,
        f"Среднедневные: {avg_daily:,.0f} €"
    )

    return [mape_card, mae_card, rmse_card, sales_card]

def _format_metric_value(value: str) -> str:
    """Форматирование значения метрики для отображения"""
    try:
        # Извлечение числовой части
        clean = value.replace("€", "").replace("%", "").replace(",", "").strip()
        numeric = float(clean)

        if "%" in value:
            return f"{numeric:.1f}%"
        elif "€" in value:
            return f"{numeric:,.0f} €".replace(",", " ")
        else:
            return f"{numeric:,.0f}".replace(",", " ")
    except (ValueError, TypeError):
        return str(value)


def _create_metric_card(title: str, value: str, color: str, description: str =None) -> html.Div:
    """Создание отдельной карточки метрики с форматированием"""
    formatted_value = _format_metric_value(value)

    children = [
        html.Div(title, className="metric-title"),
        html.Div(formatted_value, className="metric-value", style={"color": color}),
    ]
    if description:
        children.append(html.Div(description, className="metric-description"))

    return html.Div(children, className="metric-card")
