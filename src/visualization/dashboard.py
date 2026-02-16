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
        """Получение времени последнего обновления данных"""
        return self._last_update_time

    def set_last_update_time(self, timestamp: datetime = None) -> None:
        """Сохранение времени обновления данных"""
        if timestamp is None:
            timestamp = datetime.now()
        self._last_update_time = timestamp

    def _setup_layout(self) -> None:
        """Настройка layout дашборда"""
        self.app.layout = create_layout(self.data)
        self.logger.debug("Layout дашборда настроен")

    def _setup_callbacks(self) -> None:
        """Настройка callback функций для интерактивного обновления"""

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
             Input("refresh-btn", "n_clicks")]
        )
        def update_dashboard(selected_store, start_date, end_date, period_preset, n_clicks):
            """Обновление всех компонентов дашборда при изменении фильтров"""

            # Обработка period_preset для автоматической установки дат
            if period_preset and period_preset != "custom":
                end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

                if period_preset == "7days":
                    start_date = end_date - timedelta(days=7)
                elif period_preset == "30days":
                    start_date = end_date - timedelta(days=30)
                elif period_preset == "90days":
                    start_date = end_date - timedelta(days=90)
                elif period_preset == "all":
                    start_date = self.data["Date"].min()
                    end_date = self.data["Date"].max()

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
            filtered_data = self.data.copy()

            if selected_store and selected_store > 0:
                filtered_data = filtered_data[filtered_data["Store"] == selected_store]

            # Фильтрация по датам
            if start_date and end_date:
                filtered_data = filtered_data[
                    (filtered_data["Date"] >= pd.to_datetime(start_date)) &
                    (filtered_data["Date"] <= pd.to_datetime(end_date))
                ]

            # Проверка наличия данных после фильтрации
            if start_date is None or end_date is None:
                empty_fig = create_empty_chart("Выберите диапазон дат")
                last_update = f"Последнее обновление: {datetime.now().strftime("%H:%M:%S")}"
                return ["-", "-", "-", "-", empty_fig, empty_fig, empty_fig, empty_fig, "", last_update]

            # Расчет метрик
            metrics = calculate_metrics(filtered_data)
            metric_cards = create_metric_cards(metrics)

            # Создание графиков
            forecast_fig = create_forecast_chart(filtered_data)
            error_fig = create_error_distribution(filtered_data)
            store_fig = create_store_comparison(self.data, selected_store,
                                                pd.to_datetime(start_date),
                                                pd.to_datetime(end_date))
            feature_fig = create_feature_importance_chart()
            table_fig = create_data_table(filtered_data)

            last_update_str = self._format_last_update()

            return [
                metric_cards[0], metric_cards[1], metric_cards[2], metric_cards[3],
                forecast_fig, error_fig, store_fig, feature_fig, dcc.Graph(figure=table_fig, config={"displayModeBar": True}),
                last_update_str
            ]

        self.logger.debug("Callbacks дашборда настроены")

    def _format_last_update(self) -> str:
        """Формирование строки последнего обновления с указанием источника"""
        source = self.data_provider.data_source
        time_str = datetime.now().strftime("%H:%M:%S")
        return f"Последнее обновление: {time_str} (источник: {source})"

    def run(self, debug: bool = True, port: int = 8050) -> None:
        """Запуск дашборда"""
        source_info = self.data_provider.get_data_source_info()
        self.logger.info(f"Дашборд доступен по адресу http://localhost:{port}")
        self.logger.info(f"Используемые данные: магазинов={self.data['Store'].nunique()}, диапазон={self.data['Date'].min()} — {self.data['Date'].max()}, записей={len(self.data)}")
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    dashboard = ForecastingDashboard()
    dashboard.run()
