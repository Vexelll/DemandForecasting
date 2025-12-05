from dash import Dash, Input, Output, dcc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config.settings import DATA_PATH
from visualization.components.layout import create_layout
from visualization.components.metrics import calculate_metrics, create_metric_cards
from visualization.components.charts import (
    create_forecast_chart,
    create_error_distribution,
    create_store_comparison,
    create_feature_importance_chart,
    create_empty_chart,
    create_data_table
)

class ForecastingDashboard:
    def __init__(self):
        self.app = Dash(__name__)
        self.data = self.load_data()
        self.setup_layout()
        self.setup_callbacks()

    def load_data(self):
        """Загрузка данных для дашборда"""
        data_path = DATA_PATH / "outputs/predictions.csv"

        # Проверка существования файла
        if os.path.exists(data_path):
            try:
                data = pd.read_csv(data_path, parse_dates=["Date"])
                print(f"Загружено данных: {len(data)} записей")

                # Проверка необходимых колонок
                required_columns = ["Store", "Date", "PredictedSales", "ActualSales"]
                missing_columns = [col for col in required_columns if col not in data.columns]

                if missing_columns:
                    print(f"Внимание: отсутствуют колонки {missing_columns}. Используются демо-данные.")
                    return self.create_demo_data()

                return data

            except Exception as e:
                print(f"Ошибка загрузки файла: {e}. Используются демо-данные.")
                return self.create_demo_data()
        else:
            print(f"Файл не найден: {data_path}. Используются демо-данные.")
            return self.create_demo_data()

    def create_demo_data(self):
        """Создание демо-данных для тестирования"""
        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")

        # Более реалистичные демо-данные
        base_sales = 10000
        seasonal_effect = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        trend = 50 * np.arange(len(dates))
        noise = np.random.normal(0, 500, len(dates))

        actual_sales = base_sales + seasonal_effect + trend + noise
        predicted_sales = actual_sales + np.random.normal(0, 800, len(dates))

        return pd.DataFrame({
            "Store": [1] * len(dates),
            "Date": dates,
            "PredictedSales": np.maximum(predicted_sales, 0),
            "ActualSales": np.maximum(actual_sales, 0)
        })

    def setup_layout(self):
        """Настройка layout дашборда"""
        self.app.layout = create_layout(self.data)

    def setup_callbacks(self):
        """Настройка callback функций"""

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
            """Обновление всех компонентов дашборда"""

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

            # Перезагрузка данных при нажатии кнопки обновления
            if n_clicks and n_clicks > 0:
                self.data = self.load_data()
                print(f"Данные обновлены. Записей: {len(self.data)}")

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
            if len(filtered_data) == 0:
                # Возвращаем пустые графики если нет данных
                empty_metric = create_metric_cards("Нет данных", "-", "#95a5a6", "Выберите другие фильтры")
                empty_fig = create_empty_chart("Нет данных для выбранных фильтров")
                last_update = f"Последнее обновление: {datetime.now().strftime("%H:%M:%S")}"
                return [empty_metric, empty_metric, empty_metric, empty_metric, empty_fig, empty_fig, empty_fig, empty_fig, last_update]

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

            last_update = f"Последнее обновление: {datetime.now().strftime("%H:%M:%S")}"

            return [
                metric_cards[0], metric_cards[1], metric_cards[2], metric_cards[3],
                forecast_fig, error_fig, store_fig, feature_fig, dcc.Graph(figure=table_fig, config={"displayModeBar": True}),
                last_update
            ]

    def run(self, debug=True, port=8050):
        """Запуск дашборда"""
        print(f"Дашборд доступен по адресу: http://localhost:{port}")
        print("Используемые данные:")
        print(f"- Магазинов: {self.data["Store"].nunique()}")
        print(f"- Диапазон дат: {self.data["Date"].min()} до {self.data["Date"].max()}")
        print(f"- Записей: {len(self.data)}")
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    dashboard = ForecastingDashboard()
    dashboard.run()