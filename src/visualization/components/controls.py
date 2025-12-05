from dash import dcc, html
from datetime import datetime, timedelta


def create_controls(data):
    """Создание элементов управления"""
    if len(data) == 0:
        store_options = [{"label": "Нет данных", "value": 0}]
        min_date = max_date = datetime.now().date()
    else:
        store_options = [{"label": f"Магазин {i}", "value": i}
                         for i in sorted(data["Store"].unique())]
        store_options.insert(0, {"label": "Все магазины", "value": 0})

        min_date = data["Date"].min().date()
        max_date = data["Date"].max().date()

    # Предустановленные периоды
    preset_periods = [
        {"label": "Последние 7 дней", "value": "7days"},
        {"label": "Последние 30 дней", "value": "30days"},
        {"label": "Последние 90 дней", "value": "90days"},
        {"label": "Весь период", "value": "all"},
        {"label": "Произвольный", "value": "custom"}
    ]

    return html.Div([
        html.Div([
            # Выбор магазина
            html.Div([
                html.Label("Выберите магазин:", className="control-label"),
                dcc.Dropdown(
                    id="store-selector",
                    options=store_options,
                    value=0 if store_options else 0,
                    clearable=False,
                    className="store-dropdown",
                    placeholder="Выберите магазин..."
                )
            ], className="control-group"),

            # Быстрый выбор периода
            html.Div([
                html.Label("Быстрый период:", className="control-label"),
                dcc.Dropdown(
                    id="period-preset",
                    options=preset_periods,
                    value="30days",
                    clearable=False,
                    className="period-dropdown"
                )
            ], className="control-group"),

            # Диапазон дат
            html.Div([
                html.Label("Диапазон дат:", className="control-label"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=max_date - timedelta(days=30),
                    end_date=max_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format="YYYY-MM-DD",
                    className="date-picker",
                    start_date_placeholder_text="Начальная дата",
                    end_date_placeholder_text="Конечная дата"
                )
            ], className="control-group"),

            # Кнопка обновления
            html.Div([
                html.Label("Действия:", className="control-label"),
                html.Button("Обновить данные",
                            id="refresh-btn",
                            n_clicks=0,
                            className="refresh-button"),
                html.Small("Последнее обновление: -",
                           id="last-update-text",
                           style={"display": "block", "marginTop": "5px", "color": "#7f8c8d"})
            ], className="control-group")

        ], className="controls-container")
    ], className="controls-section")