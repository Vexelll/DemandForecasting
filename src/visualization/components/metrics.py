import logging
from typing import Any

import numpy as np
from dash import html

logger = logging.getLogger(__name__)

def calculate_metrics(data) -> dict[str, Any]:
    """MAE, MAPE, RMSE, MedAE, R² + цветовая оценка качества"""
    # Базовые метрики по умолчанию
    default_metrics = {
        "mape": None, "mae": None, "rmse": None,
        "total_sales": None, "bias": None, "std_error": None,
        "data_points": 0
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
            mape = float(np.mean(ape[np.isfinite(ape)]))
        else:
            mape = 0.0

    # Расчет метрик
    metrics = {
        "mape": round(mape, 2) if np.isfinite(mape) else 0.0,
        "mae": round(float(np.mean(np.abs(errors))), 2),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        "total_sales": round(float(actual.sum()), 2),
        "bias": round(float(errors.mean()), 2),
        "std_error": round(float(errors.std()), 2) if len(errors) > 1 else 0.0,
        "data_points": len(valid_data)
    }

    logger.debug(f"Метрики рассчитаны: MAPE={metrics["mape"]}%, MAE={metrics["mae"]}, RMSE={metrics["rmse"]}, точек данных={metrics["data_points"]}")

    return metrics

def create_metric_cards(metrics: dict[str, Any]) -> list[html.Div]:
    """Dash html.div карточки: значение + подпись + цвет по порогу"""
    no_data = metrics.get("data_points", 0) == 0

    # MAPE
    if no_data or metrics["mape"] is None:
        mape_color = "#95a5a6"
        mape_value = "—"
        mape_desc = "Нет данных для расчёта"
    else:
        # Цвет зависит от качества прогноза
        if metrics["mape"] < 10:
            mape_color = "#27ae60"  # отличное качество
        elif metrics["mape"] < 15:
            mape_color = "#f39c12"  # хорошее качество
        else:
            mape_color = "#e74c3c"  # требует улучшения
        mape_value = f"{metrics["mape"]:.2f}%"
        mape_desc = f"На основе {metrics["data_points"]:,} наблюдений"

    mape_card = _create_metric_card(
        "Точность прогноза (MAPE)", mape_value, mape_color, mape_desc
    )

    # MAE
    mae_color = "#3498db"
    if no_data or metrics["mae"] is None:
        mae_value = "—"
        mae_desc = "Нет данных для расчёта"
    else:
        mae_value = f"{metrics["mae"]:,.0f} €"
        mae_desc = f"Смещение: {metrics["bias"]:+,.0f} €"

    mae_card = _create_metric_card(
        "Средняя абсолютная ошибка", mae_value, mae_color, mae_desc
    )

    # RMSE
    rmse_color = "#9b59b6"
    if no_data or metrics["rmse"] is None:
        rmse_value = "—"
        rmse_desc = "Нет данных для расчёта"
    else:
        rmse_value = f"{metrics["rmse"]:,.0f} €"
        rmse_desc = f"Стд. откл. ошибки: {metrics["std_error"]:,.0f} €"

    rmse_card = _create_metric_card(
        "Среднеквадратичная ошибка", rmse_value, rmse_color, rmse_desc
    )

    # Общий объем продаж
    sales_color = "#2ecc71"
    if no_data or metrics["total_sales"] is None:
        sales_value = "—"
        sales_desc = "Нет данных для расчёта"
    else:
        avg_daily = metrics["total_sales"] / max(metrics["data_points"], 1)
        sales_value = f"{metrics["total_sales"]:,.0f} €"
        sales_desc = f"Среднедневные: {avg_daily:,.0f} €"

    sales_card = _create_metric_card(
        "Общий объём продаж", sales_value, sales_color, sales_desc
    )

    return [mape_card, mae_card, rmse_card, sales_card]

def _create_metric_card(title: str, value: str, color: str, description: str = None) -> html.Div:
    """Одна карточка: заголовок + значение + цветовая полоска"""
    children = [
        html.Div(title, className="metric-title"),
        html.Div(value, className="metric-value", style={"color": color}),
    ]
    if description:
        children.append(html.Div(description, className="metric-description"))

    return html.Div(children, className="metric-card")
