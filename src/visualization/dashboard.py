import dash
from dash import Dash, Input, Output, dcc
import pandas as pd
import logging
from datetime import datetime, timedelta
from components.layout import create_layout
from components.metrics import calculate_metrics, create_metric_cards
from components.charts import (
    create_forecast_chart,
    create_error_distribution,
    create_store_comparison,
    create_feature_importance_chart,
    create_empty_chart,
    create_data_table
)
from src.database.dashboard_data_provider import DashboardDataProvider
from config.settings import get_dashboard_config, setup_logging

class ForecastingDashboard:
    # Минимальный интервал между обновлениями данных (секунды)
    REFRESH_COOLDOWN_SECONDS = 5

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.app = Dash(__name__)
        self.data_provider = DashboardDataProvider()
        self.data = self.data_provider.load_predictions()
        self._last_update_time = datetime.now()
        self._setup_layout()
        self._setup_callbacks()

        # Логирование источника данных
        source_info = self.data_provider.get_data_source_info()
        self.logger.info(f"Дашборд инициализирован. Источник данных: {source_info["source"]}")

    def get_last_update_time(self) -> datetime:
        """Timestamp последнего обновления (для footer)"""
        return self._last_update_time

    def set_last_update_time(self, timestamp: datetime = None) -> None:
        """Ставит timestamp = now"""
        if timestamp is None:
            timestamp = datetime.now()
        self._last_update_time = timestamp

    def _setup_layout(self) -> None:
        """Собирает layout из controls + charts + footer"""
        self.app.layout = create_layout(self.data)
        self.logger.debug("Layout дашборда настроен")

    def _filter_data(self, selected_store: int, start_date, end_date, period_preset: str) -> pd.DataFrame:
        """DataFrame -> subset по Store + Date range"""
        # Обработка period_preset для автоматической установки дат
        if period_preset and period_preset != "custom":
            end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

            if period_preset == "7days":
                start_date = end_dt - timedelta(days=7)
                end_date = end_dt
            elif period_preset == "30days":
                start_date = end_dt - timedelta(days=30)
                end_date = end_dt
            elif period_preset == "90days":
                start_date = end_dt - timedelta(days=90)
                end_date = end_dt
            elif period_preset == "all":
                start_date = self.data["Date"].min()
                end_date = self.data["Date"].max()

        if start_date is None or end_date is None:
            return pd.DataFrame()

        # Построение маски фильтрации
        mask = pd.Series(True, index=self.data.index)

        if selected_store and selected_store > 0:
            mask &= self.data["Store"] == selected_store

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask &= (self.data["Date"] >= start_dt) & (self.data["Date"] <= end_dt)

        return self.data.loc[mask]

    def _setup_callbacks(self) -> None:
        """Регистрация Dash callbacks: фильтры -> графики"""

        @self.app.callback(
            [Output("mape-metric", "children"),
             Output("mae-metric", "children"),
             Output("rmse-metric", "children"),
             Output("total-sales-metric", "children"),
             Output("forecast-chart", "figure"),
             Output("error-distribution", "figure"),
             Output("store-comparison", "figure"),
             Output("feature-importance", "figure"),
             Output("data-table", "children"),
             Output("last-update-text", "children")],
            [Input("store-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("period-preset", "value"),
             Input("refresh-btn", "n_clicks")],
            prevent_initial_call=False
        )
        def update_dashboard(selected_store, start_date, end_date, period_preset, n_clicks):
            """Главный callback: пересчет всех графиков и метрик"""
            ctx = dash.callback_context

            # Проверяем, что событие пришло именно от кнопки обновления
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

                if trigger_id == "refresh-btn":
                    current_time = datetime.now()
                    last_update = self.get_last_update_time()
                    elapsed = (current_time - last_update).total_seconds()

                    # Не чаще чем раз в REFRESH_COOLDOWN_SECONDS
                    if elapsed < self.REFRESH_COOLDOWN_SECONDS:
                        self.logger.debug(f"Слишком частая перезагрузка данных (прошло {elapsed:.1f}с < {self.REFRESH_COOLDOWN_SECONDS}с)")
                    else:
                        self.data = self.data_provider.load_predictions()
                        self.set_last_update_time(current_time)
                        self.logger.info(f"Данные обновлены. Записей: {len(self.data)}")

            # Фильтрация данных
            filtered_data = self._filter_data(
                selected_store, start_date, end_date, period_preset
            )

            # Проверка наличия данных после фильтрации
            if len(filtered_data) == 0:
                empty_fig = create_empty_chart("Нет данных для выбранного периода")
                last_update = f"Обновлено: {datetime.now().strftime("%H:%M:%S")}"
                default_card = dash.html.Div([
                    dash.html.Div("—", className="metric-value", style={"color": "#95a5a6"}),
                ], className="metric-card")
                return [
                    default_card, default_card, default_card, default_card,
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    "", last_update
                ]

            # Расчёт метрик
            metrics = calculate_metrics(filtered_data)
            metric_cards = create_metric_cards(metrics)

            # Определение дат для сравнения магазинов
            start_dt = pd.to_datetime(start_date) if start_date else filtered_data["Date"].min()
            end_dt = pd.to_datetime(end_date) if end_date else filtered_data["Date"].max()

            # Создание графиков
            forecast_fig = create_forecast_chart(filtered_data)
            error_fig = create_error_distribution(filtered_data)
            store_fig = create_store_comparison(self.data, selected_store, start_dt, end_dt)
            feature_fig = create_feature_importance_chart()

            # Таблица данных (возвращаем как dcc.Graph - children для data-table)
            table_fig = create_data_table(filtered_data)
            table_component = dcc.Graph(
                figure=table_fig,
                config={"displayModeBar": False}
            )

            last_update_str = self._format_last_update()

            return [
                metric_cards[0], metric_cards[1], metric_cards[2], metric_cards[3],
                forecast_fig, error_fig, store_fig, feature_fig, table_component,
                last_update_str
            ]

        self.logger.debug("Callbacks дашборда настроены")

    def _format_last_update(self) -> str:
        """Footer текст: источники (БД/демо) + время"""
        source = self.data_provider.data_source
        time_str = datetime.now().strftime("%H:%M:%S")
        return f"Последнее обновление: {time_str} (источник: {source})"

    def run(self, debug: bool = None, port: int = None) -> None:
        """app.run_server с портом из конфига"""
        dashboard_cfg = get_dashboard_config()
        if debug is None:
            debug = dashboard_cfg.get("debug", True)
        if port is None:
            port = dashboard_cfg.get("port", 8050)

        source_info = self.data_provider.get_data_source_info()
        self.logger.info(f"Дашборд доступен по адресу http://localhost:{port}")
        self.logger.info(f"Используемые данные: магазинов={self.data["Store"].nunique()}, диапазон={self.data["Date"].min()} — {self.data["Date"].max()}, записей={len(self.data)}")
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    # Настройка логирования
    setup_logging()

    dashboard = ForecastingDashboard()
    dashboard.run()
