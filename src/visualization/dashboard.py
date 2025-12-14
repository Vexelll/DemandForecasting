import dash
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
        self._last_update_time = datetime.now()
        self.setup_layout()
        self.setup_callbacks()

    def get_last_update_time(self):
        """Получение времени последнего обновления"""
        return self._last_update_time

    def set_last_update_time(self, timestamp = None):
        """Сохранение времени обновления"""
        if timestamp is None:
            timestamp = datetime.now()
        self._last_update_time = timestamp

    def load_data(self):
        """Загрузка данных для дашборда"""
        data_path = DATA_PATH / "outputs/predictions.csv"

        # Проверка существования файла и не пустой ли он
        if os.path.exists(data_path) and data_path.stat().st_size > 0:
            try:
                data = pd.read_csv(data_path, parse_dates=["Date"])

                if len(data) == 0:
                    print("Файл с данными пуст. Используются демо-данные.")

                # Проверка необходимых колонок
                required_columns = ["Store", "Date", "PredictedSales", "ActualSales"]
                missing_columns = [col for col in required_columns if col not in data.columns]

                if missing_columns:
                    print(f"Внимание: отсутствуют колонки {missing_columns}. Используются демо-данные.")
                    return self.create_demo_data()

                print(f"Данные успешно загружены: {len(data)} записей")
                return data

            except pd.errors.EmptyDataError:
                print("Ошибка: файл данных пуст. Используются демо-данные.")
                return self.create_demo_data()
            except Exception as e:
                print(f"Ошибка загрузки файла: {e}. Используются демо-данные.")
                return self.create_demo_data()
        else:
            print(f"Файл не найден или пуст: {data_path}. Используются демо-данные.")
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

            ctx = dash.callback_context

            # Проверяем, что событие пришло именно от кнопки обновления
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

                if trigger_id == "refresh-btn":
                    # Проверяем, не слишком ли часто обновляем
                    current_time = datetime.now()
                    last_update = self.get_last_update_time()

                    if (current_time - last_update).total_seconds() < 5: # Не чаще чем раз в 5 секунд
                        print("Предупреждение: слишком частая перезагрузка данных")
                    else:
                        self.data = self.load_data()
                        self.set_last_update_time(current_time)
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
            if start_date is None or end_date is None:
                # Возвращаем пустые графики
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