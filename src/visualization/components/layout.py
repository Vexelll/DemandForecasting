from dash import dcc, html
from datetime import datetime
from .controls import create_controls


def create_layout(data):
    """Создание основного layout дашборда"""
    return html.Div([
        # Заголовок
        _create_header(),

        # Индикатор загрузки
        dcc.Loading(
            id="loading-indicator",
            type="circle",
            children=[
                # Контролы
                create_controls(data),

                # Метрики
                html.Div([
                    html.Div(id="mape-metric", className="metric-container"),
                    html.Div(id="mae-metric", className="metric-container"),
                    html.Div(id="rmse-metric", className="metric-container"),
                    html.Div(id="total-sales-metric", className="metric-container"),
                ], className="metrics-grid"),

                # Основные графики
                html.Div([
                    html.Div([
                        dcc.Graph(id="forecast-chart", config={"displayModeBar": True})
                    ], className="chart-container"),

                    html.Div([
                        dcc.Graph(id="error-distribution", config={"displayModeBar": True})
                    ], className="chart-container"),
                ], className="charts-row"),

                # Дополнительные графики
                html.Div([
                    html.Div([
                        dcc.Graph(id="store-comparison", config={"displayModeBar": True})
                    ], className="chart-container"),

                    html.Div([
                        dcc.Graph(id="feature-importance", config={"displayModeBar": True})
                    ], className="chart-container"),
                ], className="charts-row"),

                # Таблица данных
                html.Div([
                    html.H3("Детальные данные", className="section-title"),
                    html.Div(id="data-table", className="table-container")
                ], className="section"),
            ]
        ),

        # Скрытый компонент для хранения времени последнего обновления
        html.Div(id="last-update", style={"display": "none"}),

        # Футер
        _create_footer()
    ], className="dashboard-container")


def _create_header():
    """Создание заголовка дашборда"""
    return html.Div([
        html.H1("Demand Forecasting Dashboard", className="main-title"),
        html.P("Система прогнозирования спроса для розничной сети Rossmann",
               className="subtitle"),
        html.Hr(className="header-divider")
    ], className="header")


def _create_footer():
    """Создание футера дашборда"""
    return html.Div([
        html.Hr(className="footer-divider"),
        html.P([
            "Дипломный проект • ",
            html.Span("Система прогнозирования спроса", className="highlight"),
            " • ",
            html.Span(datetime.now().year, className="year")
        ], className="footer-text")
    ], className="footer")
