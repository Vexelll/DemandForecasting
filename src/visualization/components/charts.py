import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import logging
from config.settings import REPORTS_PATH

logger = logging.getLogger(__name__)

# Единая цветовая палитра дашборда
COLORS = {
    "actual": "#27ae60",        # Фактические продажи - зеленый
    "actual_fill": "rgba(39, 174, 96, 0.08)",
    "predicted": "#e74c3c",      # Прогноз модели - красный
    "error_fill": "rgba(231, 76, 60, 0.08)",
    "histogram": "#3498db",     # Гистограмма - синий
    "bar_primary": "#2ecc71",   # Столбцы - зеленый
    "bar_secondary": "#e74c3c", # Вторичная ось - красный
    "highlight": "#f1c40f",
    "mean_line": "#27ae60",     # Среднее - зеленый
    "median_line": "#f39c12",   # Медиана - оранжевый
    "zero_line": "#e74c3c",     # Нулевая ошибка - красный
   "text_primary": "#2c3e50",   # Основной текст
   "text_secondary": "#7f8c8d", # Вторичный текст
   "grid": "rgba(189, 195, 199, 0.3)",  # Сетка - светло-серая
   "border": "#bdc3c7",         # Границы
   "table_header": "#2c3e50",   # Заголовок таблицы
   "annotation_bg": "rgba(255, 255, 255, 0.92)"
}

# Единые параметры шаблона для всех графиков
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif", size=12),
    margin=dict(l=60, r=60, t=80, b=60),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
    )
)

# Градиентная палитра Viribis для горизонтальных bar chart
VIRIDIS_PALETTE = px.colors.sequential.Viridis


def _apply_common_layout(fig: go.Figure, **kwargs) -> None:
    """Применение общих параметров layout ко всем графикам"""
    merged = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**merged)

def create_forecast_chart(data):
    """График фактических vs прогнозируемых продаж"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    required_cols = ["Date", "ActualSales", "PredictedSales"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return create_empty_chart(f"Отсутствуют данные: {", ".join(missing_cols)}")

    fig = go.Figure()

    # Сортировка по дате
    data = data.sort_values("Date").copy()

    # Фактические продажи
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["ActualSales"],
        mode="lines+markers",
        name="Фактические продажи",
        line=dict(color=COLORS["actual"], width=2.5),
        marker=dict(size=4, symbol="circle"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Факт: %{y:,.0f} €<extra></extra>",
        fill="tozeroy",
        fillcolor=COLORS["actual_fill"]
    ))

    # Прогноз модели
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["PredictedSales"],
        mode="lines+markers",
        name="Прогноз модели",
        line=dict(color=COLORS["predicted"], width=2, dash="dash"),
        marker=dict(size=3, symbol="diamond"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Прогноз: %{y:,.0f} €<extra></extra>"
    ))

    # Область ошибки (доверительный интервал)
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
            fillcolor=COLORS["error_fill"],
            showlegend=False,
            hoverinfo="skip"
        ))

    # Расчет метрик для подзаголовка
    errors = data["ActualSales"] - data["PredictedSales"]
    with np.errstate(divide = "ignore", invalid ="ignore"):
        mape = np.mean(np.abs(errors) / np.maximum(data["ActualSales"], 1)) * 100
    rmse = np.sqrt(np.mean(errors ** 2))

    _apply_common_layout(
        fig,
        title=dict(
            text=f"Фактические vs Прогнозируемые продажи<br><sub style='color:{COLORS["text_secondary"]}'>MAPE: {mape:.1f}%  |  RMSE: {rmse:,.0f} €</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Дата",
            tickformat="%d.%m.%Y",
            gridcolor=COLORS["grid"],
            showgrid=True
        ),
        yaxis=dict(
            title="Продажи, €",
            tickformat=",",
            gridcolor=COLORS["grid"],
            showgrid=True
        ),
        hovermode = "x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        height=450
    )

    return fig

def create_error_distribution(data: pd.DataFrame) -> go.Figure:
    """Распределение ошибок прогнозирования"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    errors = data["ActualSales"] - data["PredictedSales"]

    # Расчет процентных ошибок
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(
            data["ActualSales"] > 0,
            (errors / data["ActualSales"] * 100),
            0 # Если продажи 0, ошибка = 0%
        )

    error_pct = error_pct[np.isfinite(error_pct)]

    if len(error_pct) == 0:
        return create_empty_chart("Нет данных для анализа ошибок")

    fig = go.Figure()

    # Гистограмма ошибок
    fig.add_trace(go.Histogram(
        x=error_pct,
        nbinsx=min(40, max(10, len(error_pct) // 5)),
        name="Распределение ошибок",
        marker=dict(
            color=COLORS["histogram"],
            opacity=0.75,
            line=dict(color="rgba(52, 152, 219, 0.9)", width=0.5)
        ),
        hovertemplate="Ошибка: %{x:.1f}%<br>Частота: %{y}<extra></extra>"
    ))

    # Вертикальная линия нулевой ошибки
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLORS["zero_line"],
        line_width=1.5,
        annotation_text="Нулевая ошибка",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color=COLORS["zero_line"]
    )

    # Статистика ошибок
    mean_error_pct = float(np.mean(error_pct))
    median_error_pct = float(np.median(error_pct))
    std_error_pct = float(np.std(error_pct))
    min_error_pct = float(np.min(error_pct))
    max_error_pct = float(np.max(error_pct))

    # Линия среднего
    fig.add_vline(
        x=mean_error_pct,
        line_dash="dot",
        line_color=COLORS["mean_line"],
        line_width=1.5,
        annotation_text=f"Среднее: {mean_error_pct:.1f} €",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=COLORS["mean_line"]
    )

    # Линия медианы
    fig.add_vline(
        x=median_error_pct,
        line_dash="dot",
        line_color=COLORS["median_line"],
        line_width=1.5,
        annotation_text=f"Медиана: {median_error_pct:.1f} €",
        annotation_font_size=11,
        annotation_font_color=COLORS["median_line"]
    )

    _apply_common_layout(
        fig,
        title=dict(
            text="Распределение ошибок прогнозирования",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Ошибка (%)",
            gridcolor=COLORS["grid"],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Частота",
            gridcolor=COLORS["grid"],
            showgrid=True
        ),
        height=500,
        bargap=0.05,
        showlegend=False,
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # Аннотация со сводной статистикой
    stats_text = f"""
        <b>Статистика ошибок:</b><br>
        • Среднее: {mean_error_pct:.1f} €<br>
        • Медиана: {median_error_pct:.1f} €<br>
        • Стд. откл.: {std_error_pct:.1f} €<br>
        • Мин / Макс: {min_error_pct:.1f} € / {max_error_pct:.1f} €<br>
        • Наблюдений: {len(error_pct):,}
        """

    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        font=dict(size=11, color=COLORS["text_primary"]),
        bgcolor=COLORS["annotation_bg"],
        bordercolor=COLORS["border"],
        borderwidth=1,
        borderpad=10
    )

    return fig

def create_store_comparison(full_data: pd.DataFrame, selected_store: int, start_date: pd.Timestamp, end_date: pd.Timestamp) -> go.Figure:
    """Сравнение магазинов по средним продажам"""
    # Проверяем, что даты не None
    if start_date is None or end_date is None:
        return create_empty_chart("Выберите диапазон дат для сравнения")

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except Exception as e:
        logger.error(f"Ошибка парсинга дат для сравнения магазинов: {e}")
        return create_empty_chart(f"Ошибка в формате дат: {e}")

    filtered_data = full_data[
        (full_data["Date"] >= start_date) &
        (full_data["Date"] <= end_date)
    ]

    if len(filtered_data) == 0:
        return create_empty_chart("Нет данных для сравнения")

    grouped = filtered_data.groupby("Store").agg({
        "ActualSales": ["mean", "std", "sum"],
        "PredictedSales": "mean"
    })

    # Преобразование
    store_stats = pd.DataFrame({
        "AvgSales": grouped[("ActualSales", "mean")],
        "StdSales": grouped[("ActualSales", "std")],
        "TotalSales": grouped[("ActualSales", "sum")],
        "AvgPredicted": grouped[("PredictedSales", "mean")]
    }).round(2)

    store_stats = store_stats.sort_values("AvgSales", ascending=False).head(15)

    if len(store_stats) == 0:
        return create_empty_chart("Нет данных для сравнения магазинов")

    # Расчет MAPE для каждого магазина
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
        marker=dict(
            color=COLORS["bar_primary"],
            opacity=0.85,
            line=dict(color="rgba(46, 204, 113, 0.9)", width=0.5)
        ),
        error_y=dict(
            type="data",
            array=store_stats["StdSales"],
            visible=True,
            color="rgba(0,0,0,0.2)",
            thickness=1.5
        ),
        hovertemplate=(
            "<b>Магазин %{x}</b><br>"
            "Средние продажи: %{y:,.0f} €<br>"
            "Std: %{customdata[0]:,.0f} €<br>"
            "MAPE: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=np.column_stack([store_stats["StdSales"], store_stats["Accuracy"]])
    ))

    # Линия прогнозируемых продаж
    fig.add_trace(go.Scatter(
        x=store_stats.index.astype(str),
        y=store_stats["AvgPredicted"],
        mode="lines+markers",
        name="Прогноз (средний)",
        line=dict(color=COLORS["bar_secondary"], width=2),
        marker=dict(size=7, symbol="diamond"),
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
                color=COLORS["highlight"],
                line=dict(width=2, color="black")
            ),
            name="Выбранный магазин",
            hovertemplate=(
                "<b>Магазин %{x} (выбран)</b><br>"
                "Продажи: %{y:,.0f} €<br>"
                "MAPE: %{customdata:.1f}%<extra></extra>"
            ),
            customdata=[store_stats.loc[selected_store, "Accuracy"]]
        ))

    _apply_common_layout(
        fig,
        title=dict(
            text="Топ-15 магазинов по средним продажам",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Магазин",
            type="category",
            tickangle=-45
        ),
        yaxis=dict(
            title="Средние продажи (€)",
            tickformat=",",
            gridcolor=COLORS["grid"]
        ),
        yaxis2=dict(
            title="Прогнозируемые продажи (€)",
            tickformat=",",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=500,
        barmode="group",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_feature_importance_chart() -> go.Figure:
    """График важности признаков модели LightGBM"""
    importance_path = REPORTS_PATH / "feature_importance_top20.csv"

    if os.path.exists(importance_path):
        try:
            # Загрузка реальных данных важности признаков
            importance_df = pd.read_csv(importance_path)

            # Сортировка и выбор топ-20
            importance_df = importance_df.sort_values("importance", ascending=True).tail(20)

            fig = go.Figure()

            # Нормализация для цветовой шкалы
            norm_importance = (
                (importance_df["importance"] - importance_df["importance"].min()) /
                max(importance_df["importance"].max() - importance_df["importance"].min(), 1e-9)
            )
            bar_colors = [
                VIRIDIS_PALETTE[int(v * (len(VIRIDIS_PALETTE) - 1))] for v in norm_importance
            ]

            fig.add_trace(go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0.3, color="rgba(0, 0, 0, 0.15)")
                ),
                hovertemplate="<b>%{y}</b><br>Важность: %{x:.4f}<extra></extra>"
            ))

            total_importance = importance_df["importance"].sum()

            _apply_common_layout(
                fig,
                title=dict(
                    text=(
                        f"Топ-20 важнейших признаков модели"
                        f"<br><sub style='color:{COLORS["text_secondary"]}'>"
                        f"Суммарная важность: {total_importance:.3f}</sub>"
                    ),
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Важность признака",
                    gridcolor=COLORS["grid"]
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(size=10)
                ),
                height=550,
                margin=dict(l=160, r=60, t=80, b=60),
                showlegend=False
            )

            logger.info(f"График важности признаков построен: {len(importance_df)} признаков из {importance_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка загрузки важности признаков: {e}")

    # Если файл не найден или ошибка - создаем информационный график
    logger.warning(f"Файл важности признаков не найден: {importance_path}. График будет доступен после обучения модели")
    return create_info_chart(
        "Важность признаков модели",
        "График будет доступен после обучения модели.<br>"
        "Файл lgbm_feature_importance.csv не найден."
    )

def create_sales_trend_chart(data: pd.DataFrame) -> go.Figure:
    """График тренда продаж со скользящим средним"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для анализа тренда")

    # Группировка по дате
    daily_sales = data.groupby("Date")["ActualSales"].sum().reset_index()
    daily_sales = daily_sales.sort_values("Date")

    fig = go.Figure()

    # Линия фактических продаж
    fig.add_trace(go.Scatter(
        x=daily_sales["Date"],
        y=daily_sales["ActualSales"],
        mode="lines",
        name="Фактические продажи",
        line=dict(color=COLORS["histogram"], width=1.5),
        opacity=0.6,
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Продажи: %{y:,.0f} €<extra></extra>"
    ))

    # Скользящее среднее (7 дней)
    daily_sales["MovingAvg"] = daily_sales["ActualSales"].rolling(window=7, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=daily_sales["Date"],
        y=daily_sales["MovingAvg"],
        mode="lines",
        name="Скользящее среднее (7 дней)",
        line=dict(color=COLORS["predicted"], width=2.5),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Среднее (7д): %{y:,.0f} €<extra></extra>"
    ))

    # Расчет статистики тренда
    total_sales = daily_sales["ActualSales"].sum()
    avg_daily = daily_sales["ActualSales"].mean()
    growth_rate = 0.0
    if len(daily_sales) > 1 and daily_sales["ActualSales"].iloc[0] > 0:
        growth_rate = ((daily_sales["ActualSales"].iloc[-1] / daily_sales["ActualSales"].iloc[0]) - 1) * 100

    _apply_common_layout(
        fig,
        title=dict(
            text=(
                f"Тренд продаж"
                f"<br><sub style='color:{COLORS["text_secondary"]}'>"
                f"Объём: {total_sales:,.0f} €  |  "
                f"Среднедневные: {avg_daily:,.0f} €  |  "
                f"Рост: {growth_rate:.1f}%</sub>"
            ),
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Дата",
            tickformat="%d.%m.%Y",
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title="Продажи, €",
            tickformat=",",
            gridcolor=COLORS["grid"]
        ),
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_data_table(data: pd.DataFrame) -> go.Figure:
    """Таблица с детальными данными прогнозов"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для таблицы")

    # Выбираем нужные колонки и сортируем
    table_data = data[["Store", "Date", "ActualSales", "PredictedSales"]].copy()

    # Добавляем расчетные колонки
    table_data["Error"] = table_data["ActualSales"] - table_data["PredictedSales"]
    table_data["ErrorPct"] = (table_data["Error"] / table_data["ActualSales"].replace(0, np.nan) * 100).round(2)

    # Сортировка по дате и магазину
    table_data = table_data.sort_values(["Date", "Store"])

    # Ограничиваем количество строк для производительности
    total_rows = len(table_data)
    table_data = table_data.head(1000)

    # Цвет ячеек ошибки: зеленый - недопрогноз (факт > прогноз), красный - перепрогноз
    error_colors = [
        "rgba(39, 174, 96, 0.1)" if val >= 0 else "rgba(231, 76, 60, 0.1)" for val in table_data["Error"]
    ]

    # Создаем таблицу
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Магазин</b>", "<b>Дата</b>", "<b>Факт (€)</b>", "<b>Прогноз (€)</b>", "<b>Ошибка (€)</b>", "<b>Ошибка (%)</b>"],
            fill_color=COLORS["table_header"],
            align="center",
            font=dict(color="white", size=12, family="Segoe UI"),
            height=40,
            line_color="white"
        ),
        cells=dict(
            values=[
                table_data["Store"],
                table_data["Date"].dt.strftime("%d.%m.%Y"),
                table_data["ActualSales"].round(0),
                table_data["PredictedSales"].round(0),
                table_data["Error"].round(0),
                table_data["ErrorPct"]
            ],
            fill_color=["white", "white", "white", "white", error_colors, error_colors],
            align="center",
            font=dict(color=COLORS["text_primary"], size=11),
            height=30,
            format=[None, None, ",.0f", ",.0f", "+,.0f", "+.2f"],
            line_color=COLORS["grid"]
        )
    )])

    displayed = len(table_data)
    title_suffix = f"(показано {displayed:,} из {total_rows:,})" if total_rows > 1000 else ""

    fig.update_layout(
        title=dict(
            text=f"Детальные данные прогнозов{title_suffix}",
            x=0.5,
            font=dict(size=14, color=COLORS["text_primary"])
        ),
        height=min(500, 120 + len(table_data) * 30),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_info_chart(title: str, message: str) -> go.Figure:
    """Создание информационного графика с сообщением"""
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["text_secondary"]),
        align="center"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return fig


def create_empty_chart(message: str) -> go.Figure:
    """Создание пустого графика с сообщением"""
    return create_info_chart("Нет данных", message)
