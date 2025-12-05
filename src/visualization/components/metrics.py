from dash import html
import numpy as np


def calculate_metrics(data):
    """Расчет метрик качества прогнозирования с проверкой данных"""
    if len(data) == 0:
        return {
            "mape": 0, "mae": 0, "rmse": 0,
            "total_sales": 0, "bias": 0, "std_error": 0,
            "data_points": 0
        }

    actual = data["ActualSales"]
    predicted = data["PredictedSales"]

    # Проверка наличия данных
    if len(actual) == 0 or len(predicted) == 0:
        return {
            "mape": 0, "mae": 0, "rmse": 0,
            "total_sales": 0, "bias": 0, "std_error": 0,
            "data_points": 0
        }

    errors = actual - predicted

    metrics = {
        "mape": np.mean(np.abs(errors) / np.maximum(actual, 1)) * 100,
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "total_sales": actual.sum(),
        "bias": errors.mean(),
        "std_error": errors.std(),
        "data_points": len(data)
    }

    # Округление для читаемости
    metrics["mape"] = round(metrics["mape"], 2)
    metrics["mae"] = round(metrics["mae"], 2)
    metrics["rmse"] = round(metrics["rmse"], 2)
    metrics["total_sales"] = round(metrics["total_sales"], 2)
    metrics["bias"] = round(metrics["bias"], 2)
    metrics["std_error"] = round(metrics["std_error"], 2)

    return metrics


def create_metric_cards(metrics):
    """Создание карточек с метриками с улучшенным форматированием"""

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

    mape_card = create_metric_card(
        "Точность прогноза (MAPE)",
        f"{metrics["mape"]:.2f}%",
        mape_color,
        f"На основе {metrics["data_points"]} точек данных"
    )

    mae_card = create_metric_card(
        "Средняя абсолютная ошибка",
        f"{metrics["mae"]:,.0f} €",
        mae_color,
        f"Смещение: {metrics["bias"]:,.0f} €"
    )

    rmse_card = create_metric_card(
        "Среднеквадратичная ошибка",
        f"{metrics["rmse"]:,.0f} €",
        rmse_color,
        f"Std ошибки: {metrics["std_error"]:,.0f} €"
    )

    sales_card = create_metric_card(
        "Общий объем продаж",
        f"{metrics["total_sales"]:,.0f} €",
        sales_color,
        f"Среднедневные: {metrics["total_sales"] / max(metrics["data_points"], 1):,.0f} €"
    )

    return [mape_card, mae_card, rmse_card, sales_card]


def create_metric_card(title, value, color, description=None):
    """Создание отдельной карточки метрики"""
    return html.Div([
        html.Div(title, className="metric-title"),
        html.Div(value, className="metric-value", style={"color": color}),
        html.Div(description, className="metric-description") if description else None
    ], className="metric-card")