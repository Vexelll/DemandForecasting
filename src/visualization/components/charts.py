import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from config.settings import REPORTS_PATH


def create_forecast_chart(data):
    """График прогнозов vs факт с улучшенной визуализацией"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    fig = go.Figure()

    # Сортировка по дате
    data = data.sort_values("Date")

    # Фактические продажи
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["ActualSales"],
        mode="lines+markers",
        name="Фактические продажи",
        line=dict(color="#27ae60", width=3),
        marker=dict(size=6, symbol="circle"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Факт: %{y:,.0f} €<br><extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(39, 174, 96, 0.1)"
    ))

    # Прогноз модели
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["PredictedSales"],
        mode="lines+markers",
        name="Прогноз модели",
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(size=4, symbol="diamond"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Прогноз: %{y:,.0f} €<extra></extra>"
    ))

    # Область ошибки
    if "AbsoluteError" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["ActualSales"] + data["AbsoluteError"],
            mode="lines",
            name="Верхняя граница ошибки",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["ActualSales"] - data["AbsoluteError"],
            mode="lines",
            name="Нижняя граница ошибки",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(231, 76, 60, 0.1)",
            showlegend=False,
            hoverinfo="skip"
        ))

    # Расчет метрик для отображения
    errors = data["ActualSales"] - data["PredictedSales"]
    mape = np.mean(np.abs(errors) / np.maximum(data["ActualSales"], 1)) * 100
    rmse = np.sqrt(np.mean(errors ** 2))

    fig.update_layout(
        title=dict(
            text=f"Фактические vs Прогнозируемые продажи<br><sub>MAPE: {mape:.1f}% | RMSE: {rmse:,.0f} €</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Дата",
            tickformat="%d.%m.%Y",
            gridcolor="lightgray",
            showgrid=True
        ),
        yaxis=dict(
            title="Продажи, €",
            tickformat=",",
            gridcolor="lightgray",
            showgrid=True
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor="white"
    )

    return fig


def create_error_distribution(data):
    """Распределение ошибок прогнозирования с улучшенной статистикой"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    errors = data["ActualSales"] - data["PredictedSales"]
    error_pct = (errors / data["ActualSales"] * 100).round(2)

    fig = go.Figure()

    # Гистограмма ошибок
    fig.add_trace(go.Histogram(
        x=error_pct,
        nbinsx=30,
        name="Распределение ошибок",
        marker_color="#3498db",
        opacity=0.7,
        hovertemplate="Ошибка: %{x:.1f}%<br>Частота: %{y}<extra></extra>"
    ))

    # Вертикальная линия нулевой ошибки
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Нулевая ошибка",
        annotation_position="top right",
        annotation_font_size=12
    )

    # Статистика ошибок
    mean_error = errors.mean()
    std_error = errors.std()
    median_error = errors.median()
    min_error = errors.min()
    max_error = errors.max()

    # Добавление линий для статистики
    fig.add_vline(
        x=mean_error,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Средняя: {mean_error:,.0f} €",
        annotation_position="top left"
    )

    fig.add_vline(
        x=median_error,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Медиана: {median_error:,.0f} €"
    )

    fig.update_layout(
        title=dict(
            text="Распределение ошибок прогнозирования",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Процент ошибки (%)",
            gridcolor="lightgray",
            showgrid=True
        ),
        yaxis=dict(
            title="Частота",
            gridcolor="lightgray",
            showgrid=True
        ),
        template="plotly_white",
        height=500,
        bargap=0.1,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=50, r=150, t=50, b=50)
    )

    # Добавление статистики в аннотацию с фиксированной позицией
    stats_text = f"""
        <b>Статистика ошибок:</b><br>
        • Средняя: {mean_error:,.0f} €<br>
        • Медиана: {median_error:,.0f} €<br>
        • Стандартное отклонение: {std_error:,.0f} €<br>
        • Диапазон: {min_error:,.0f} € - {max_error:,.0f} €<br>
        • Количество данных: {len(errors):,}
        """

    fig.add_annotation(
        x=0.10,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#2c3e50"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#bdc3c7",
        borderwidth=1,
        borderpad=10,
        width=240
    )

    return fig


def create_store_comparison(full_data, selected_store, start_date, end_date):
    """Сравнение магазинов с улучшенной визуализацией"""
    # Проверяем что даты не None
    if start_date is None or end_date is None:
        return create_empty_chart("Выберите диапазон дат для сравнения")

    filtered_data = full_data[
        (full_data["Date"] >= start_date) &
        (full_data["Date"] <= end_date)
        ]

    if len(filtered_data) == 0:
        return create_empty_chart("Нет данных для сравнения")

    # Группировка по магазинам
    store_stats = filtered_data.groupby("Store").agg({
        "ActualSales": ["mean", "std", "sum"],
        "PredictedSales": "mean"
    }).round(2)

    store_stats.columns = ["AvgSales", "StdSales", "TotalSales", "AvgPredicted"]
    store_stats = store_stats.sort_values("AvgSales", ascending=False).head(15)

    # Расчет точности для каждого магазина
    store_accuracy = []
    for store in store_stats.index:
        store_data = filtered_data[filtered_data["Store"] == store]
        if len(store_data) > 0:
            errors = store_data["ActualSales"] - store_data["PredictedSales"]
            mape = np.mean(np.abs(errors) / np.maximum(store_data["ActualSales"], 1)) * 100
            store_accuracy.append(mape)
        else:
            store_accuracy.append(0)

    store_stats["Accuracy"] = store_accuracy

    fig = go.Figure()

    # Столбцы со средними продажами
    fig.add_trace(go.Bar(
        x=store_stats.index.astype(str),
        y=store_stats["AvgSales"],
        name="Средние продажи",
        marker_color="#2ecc71",
        error_y=dict(
            type="data",
            array=store_stats["StdSales"],
            visible=True,
            color="rgba(0,0,0,0.3)"
        ),
        hovertemplate="<b>Магазин %{x}</b><br>" +
                      "Средние продажи: %{y:,.0f} €<br>" +
                      "Std: %{customdata[0]:,.0f} €<br>" +
                      "Точность: %{customdata[1]:.1f}%<extra></extra>",
        customdata=np.column_stack([store_stats["StdSales"], store_stats["Accuracy"]])
    ))

    # Линия прогнозируемых продаж
    fig.add_trace(go.Scatter(
        x=store_stats.index.astype(str),
        y=store_stats["AvgPredicted"],
        mode="lines+markers",
        name="Прогнозируемые продажи",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=8),
        yaxis="y2",
        hovertemplate="<b>Магазин %{x}</b><br>Прогноз: %{y:,.0f} €<extra></extra>"
    ))

    # Подсветка выбранного магазина
    if selected_store and selected_store in store_stats.index:
        selected_idx = list(store_stats.index).index(selected_store)
        fig.add_trace(go.Scatter(
            x=[store_stats.index.astype(str)[selected_idx]],
            y=[store_stats.loc[selected_store, "AvgSales"]],
            mode="markers",
            marker=dict(
                size=20,
                symbol="star",
                color="gold",
                line=dict(width=2, color="black")
            ),
            name="Выбранный магазин",
            hovertemplate="<b>Магазин %{x} (выбран)</b><br>" +
                          "Продажи: %{y:,.0f} €<br>" +
                          "Точность: %{customdata:.1f}%<extra></extra>",
            customdata=[store_stats.loc[selected_store, "Accuracy"]]
        ))

    fig.update_layout(
        title=dict(
            text="Топ-15 магазинов по средним продажам",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Магазин",
            type="category",
            tickangle=45
        ),
        yaxis=dict(
            title="Средние продажи (€)",
            tickformat=",",
            gridcolor="lightgray"
        ),
        yaxis2=dict(
            title="Прогнозируемые продажи (€)",
            tickformat=",",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        height=500,
        barmode="group",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white"
    )

    return fig


def create_feature_importance_chart():
    """График важности признаков с загрузкой реальных данных"""
    importance_path = REPORTS_PATH / "lgbm_feature_importance.csv"

    if os.path.exists(importance_path):
        try:
            # Загрузка реальных данных важности признаков
            importance_df = pd.read_csv(importance_path)

            # Сортировка и выбор топ-20
            importance_df = importance_df.sort_values("importance", ascending=True).tail(20)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation="h",
                marker_color=px.colors.sequential.Viridis,
                hovertemplate="<b>%{y}</b><br>Важность: %{x:.4f}<extra></extra>"
            ))

            total_importance = importance_df["importance"].sum()

            fig.update_layout(
                title=dict(
                    text=f"Топ-20 самых важных признаков модели<br><sub>Общая важность: {total_importance:.3f}</sub>",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Важность признака",
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="Признак",
                    autorange="reversed",
                    tickfont=dict(size=10)
                ),
                template="plotly_white",
                height=500,
                margin=dict(l=150, r=50, t=80, b=50),
                plot_bgcolor="white"
            )

            return fig

        except Exception as e:
            print(f"Ошибка загрузки важности признаков: {e}")

    # Если файл не найден или ошибка - создаем информационный график
    return create_info_chart(
        "Важность признаков модели",
        "График будет доступен после обучения модели.<br>Файл lgbm_feature_importance.csv не найден."
    )


def create_sales_trend_chart(data):
    """График тренда продаж с сезонной декомпозицией"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для анализа тренда")

    # Группировка по дате
    daily_sales = data.groupby("Date")["ActualSales"].sum().reset_index()
    daily_sales = daily_sales.sort_values("Date")

    fig = go.Figure()

    # Линия тренда
    fig.add_trace(go.Scatter(
        x=daily_sales["Date"],
        y=daily_sales["ActualSales"],
        mode="lines",
        name="Фактические продажи",
        line=dict(color="#3498db", width=2),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Продажи: %{y:,.0f} €<extra></extra>"
    ))

    # Скользящее среднее (7 дней)
    daily_sales["MovingAvg"] = daily_sales["ActualSales"].rolling(window=7, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=daily_sales["Date"],
        y=daily_sales["MovingAvg"],
        mode="lines",
        name="Скользящее среднее (7 дней)",
        line=dict(color="#e74c3c", width=3),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Среднее: %{y:,.0f} €<extra></extra>"
    ))

    # Расчет статистики тренда
    total_sales = daily_sales["ActualSales"].sum()
    avg_daily = daily_sales["ActualSales"].mean()
    growth_rate = ((daily_sales["ActualSales"].iloc[-1] / daily_sales["ActualSales"].iloc[0]) - 1) * 100 if len(
        daily_sales) > 1 else 0

    fig.update_layout(
        title=dict(
            text=f"Тренд продаж<br><sub>Объем: {total_sales:,.0f} € | Среднедневные: {avg_daily:,.0f} € | Рост: {growth_rate:.1f}%</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Дата",
            tickformat="%d.%m.%Y",
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title="Продажи, €",
            tickformat=",",
            gridcolor="lightgray"
        ),
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white"
    )

    return fig


def create_data_table(data):
    """Создание таблицы с детальными данными прогнозов"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для таблицы")

    # Выбираем нужные колонки и сортируем
    table_data = data[["Store", "Date", "ActualSales", "PredictedSales"]].copy()

    # Добавляем расчетные колонки
    table_data["Error"] = table_data["ActualSales"] - table_data["PredictedSales"]
    table_data["ErrorPct"] = (table_data["Error"] / table_data["ActualSales"] * 100).round(2)

    # Сортировка по дате и магазину
    table_data = table_data.sort_values(["Date", "Store"])

    # Ограничиваем количество строк для производительности
    table_data = table_data.head(1000)

    # Создаем таблицу
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Магазин", "Дата", "Факт (€)", "Прогноз (€)", "Ошибка (€)", "Ошибка (%)"],
            fill_color="#3498db",
            align="center",
            font=dict(color="white", size=12),
            height=40
        ),
        cells=dict(
            values=[
                table_data["Store"],
                table_data["Date"].dt.strftime("%d.%m.%Y"),
                table_data["ActualSales"].round(2),
                table_data["PredictedSales"].round(2),
                table_data["Error"].round(2),
                table_data["ErrorPct"]
            ],
            fill_color="white",
            align="center",
            font=dict(color="black", size=11),
            height=30,
            format=[None, None, ",.2f", ",.2f", ",.2f", ".2f"]
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"Детальные данные прогнозов ({len(table_data)} записей)",
            x=0.5,
            font=dict(size=14)
        ),
        height=min(400, 100 + len(table_data) * 30),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_info_chart(title, message):
    """Создание информационного графика с сообщением"""
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
        align="center"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return fig


def create_empty_chart(message):
    """Создание пустого графика с сообщением"""
    return create_info_chart("Нет данных", message)