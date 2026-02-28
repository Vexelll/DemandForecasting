import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import logging
from functools import lru_cache
from config.settings import REPORTS_PATH, get_model_config, get_reporting_config

logger = logging.getLogger(__name__)

# Единая цветовая палитра дашборда
COLORS = {
    "actual": "#27ae60",        # Фактические продажи - зеленый
    "actual_fill": "rgba(39, 174, 96, 0.06)",
    "predicted": "#e74c3c",      # Прогноз модели - красный
    "error_fill": "rgba(231, 76, 60, 0.06)",
    "histogram": "#3498db",     # Гистограмма - синий
    "bar_primary": "#2ecc71",   # Столбцы - зеленый
    "bar_secondary": "#e74c3c", # Вторичная ось - красный
    "highlight": "#f1c40f",
    "mean_line": "#27ae60",     # Среднее - зеленый
    "median_line": "#f39c12",   # Медиана - оранжевый
    "zero_line": "#e74c3c",     # Нулевая ошибка - красный
    "kde_line": "#8e44ad",      # Линия плотности KDE - фиолетовый
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

# Единые параметры для горизонтальной легенды
HORIZONTAL_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    bgcolor="rgba(255, 255, 255, 0.8)"
)


def _apply_common_layout(fig: go.Figure, **kwargs) -> None:
    """Мержит LAYOUT_DEFAULTS с переданными kwargs"""
    merged = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**merged)

def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """mape без деления на ноль - нулевые Sales пропускаются"""
    with np.errstate(divide="ignore", invalid="ignore"):
        non_zero = actual > 0
        if non_zero.sum() == 0:
            return 0.0
        errors = np.abs(actual[non_zero] - predicted[non_zero])
        ape = errors / actual[non_zero] * 100
        result = float(np.mean(ape[np.isfinite(ape)]))
        return result if np.isfinite(result) else 0.0

def _add_reference_lines(fig: go.Figure, lines: list, data: np.ndarray) -> None:
    """Вертикальные линии (mean, median, zero) - с разнесением подписей, чтобы не налезали"""
    # Порог близости: 8% от видимого диапазона оси X
    data_range = float(np.max(data) - np.min(data)) if len(data) > 1 else 1.0
    proximity_threshold = data_range * 0.08

    # Уровни Y для аннотаций (в paper-координатах, 0=низ, 1=верх)
    Y_LEVELS = [0.96, 0.88, 0.80]

    # Добавляем вертикальные линии (без аннотаций)
    for line_spec in lines:
        fig.add_vline(
            x=line_spec["x"],
            line_dash=line_spec["dash"],
            line_color=line_spec["color"],
            line_width=1.5
        )

    # Сортируем аннотации по x-позиции для корректного расслоения
    sorted_lines = sorted(lines, key=lambda l: l["x"])

    # Назначаем Y-уровни с проверкой коллизий
    assigned_levels = []
    for i, line_spec in enumerate(sorted_lines):
        level = 0  # начинаем с верхнего уровня

        # Проверяем столкновение с уже размещёнными аннотациями
        for prev_idx, prev_level in assigned_levels:
            prev_x = sorted_lines[prev_idx]["x"]
            distance = abs(line_spec["x"] - prev_x)

            # Если x-позиции близки и на том же Y-уровне -> опускаем
            if distance < proximity_threshold and level == prev_level:
                level += 1

        # Ограничение: не более 3 уровней
        level = min(level, len(Y_LEVELS) - 1)
        assigned_levels.append((i, level))

        y_pos = Y_LEVELS[level]

        # Стрелка нужна, если аннотация сдвинута вниз (уровень > 0)
        show_arrow = level > 0

        fig.add_annotation(
            x=line_spec["x"],
            y=y_pos,
            xref="x",
            yref="paper",
            text=f"<b>{line_spec['label']}</b>",
            showarrow=show_arrow,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=line_spec["color"],
            ax=0,
            ay=25 if show_arrow else 0,
            font=dict(size=11, color=line_spec["color"]),
            bgcolor=COLORS["annotation_bg"],
            bordercolor=line_spec["color"],
            borderwidth=1,
            borderpad=4
        )


def create_forecast_chart(data):
    """Основной график: два trace (факт + прогноз) + метрики в аннотации"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    required_cols = ["Date", "ActualSales", "PredictedSales"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return create_empty_chart(f"Отсутствуют данные: {", ".join(missing_cols)}")

    # Агрегация по дате (если несколько магазинов - показываем сумму)
    num_stores = data["Store"].nunique()
    if num_stores > 1:
        plot_data = (
            data.groupby("Date", as_index=False)
            .agg(ActualSales=("ActualSales", "sum"),
                PredictedSales=("PredictedSales", "sum"))
            .sort_values("Date")
        )
        subtitle_prefix = f"Агрегация по {num_stores} магазинам"
    else:
        plot_data = data[["Date", "ActualSales", "PredictedSales"]].sort_values("Date")
        store_id = data["Store"].iloc[0]
        subtitle_prefix = f"Магазин {store_id}"

    # Адаптивный режим: маркеры только при малом количестве точек
    num_points = len(plot_data)
    mode = "lines+markers" if num_points <= 90 else "lines"
    marker_size = 4 if num_points <= 90 else 0

    fig = go.Figure()

    # Фактические продажи
    fig.add_trace(go.Scatter(
        x=plot_data["Date"],
        y=plot_data["ActualSales"],
        mode=mode,
        name="Фактические продажи",
        line=dict(color=COLORS["actual"], width=2.5),
        marker=dict(size=marker_size, symbol="circle"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Факт: %{y:,.0f} €<extra></extra>",
        fill="tozeroy",
        fillcolor=COLORS["actual_fill"]
    ))

    # Прогноз модели
    fig.add_trace(go.Scatter(
        x=plot_data["Date"],
        y=plot_data["PredictedSales"],
        mode=mode,
        name="Прогноз модели",
        line=dict(color=COLORS["predicted"], width=2, dash="dash"),
        marker=dict(size=max(marker_size - 1, 0), symbol="diamond"),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Прогноз: %{y:,.0f} €<extra></extra>"
    ))

    # Расчет метрик для подзаголовка
    actual_vals = plot_data["ActualSales"].values
    predicted_vals = plot_data["PredictedSales"].values
    mape = _safe_mape(actual_vals, predicted_vals)
    errors = actual_vals - predicted_vals
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    _apply_common_layout(
        fig,
        title=dict(
            text=f"Фактические vs Прогнозируемые продажи<br><sub style='color:{COLORS["text_secondary"]}'>{subtitle_prefix}  |  MAPE: {mape:.1f}%  |  RMSE: {rmse:,.0f} €</sub>",
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
        legend=HORIZONTAL_LEGEND,
        height=450
    )

    return fig

def create_error_distribution(data: pd.DataFrame) -> go.Figure:
    """Гистограмма ошибок + KDE + вертикальные линии mean/median/zero"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для отображения")

    errors = data["ActualSales"].values - data["PredictedSales"].values
    actual = data["ActualSales"].values

    # Расчет процентных ошибок (только для ненулевых продаж)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual > 0, (errors / actual) * 100, np.nan)

    error_pct_raw  = error_pct[np.isfinite(error_pct)]

    if len(error_pct_raw) == 0:
        return create_empty_chart("Нет данных для анализа ошибок")

    n_total = len(error_pct_raw)

    # Статистика по полной выборке (до обрезки)
    mean_err = float(np.mean(error_pct_raw))
    median_err = float(np.median(error_pct_raw))
    std_err = float(np.std(error_pct_raw))
    min_err = float(np.min(error_pct_raw))
    max_err = float(np.max(error_pct_raw))

    # Перцентильная обрезка выбросов для оси X
    p1, p99 = np.percentile(error_pct_raw, [1, 99])
    clip_mask = (error_pct_raw >= p1) & (error_pct_raw <= p99)
    error_pct_display = error_pct_raw[clip_mask]
    n_clipped = n_total - len(error_pct_display)

    # Если обрезка удалила слишком много (>10%) - ослабляем до 0.5–99.5
    if n_clipped > n_total * 0.10:
        p_lo, p_hi = np.percentile(error_pct_raw, [0.5, 99.5])
        clip_mask = (error_pct_raw >= p_lo) & (error_pct_raw <= p_hi)
        error_pct_display = error_pct_raw[clip_mask]
        n_clipped = n_total - len(error_pct_display)

    # Для малых выборок (<100) обрезку не применяем - каждая точка важна
    if n_total < 100:
        error_pct_display = error_pct_raw
        n_clipped = 0

    n_bins = _compute_adaptive_bins(error_pct_display)

    fig = go.Figure()

    # Гистограмма (нормализованная к плотности)
    fig.add_trace(go.Histogram(
        x=error_pct_display,
        nbinsx=n_bins,
        histnorm="probability density",
        name="Гистограмма",
        marker=dict(
            color=COLORS["histogram"],
            opacity=0.7,
            line=dict(color="rgba(52, 152, 219, 0.9)", width=0.5)
        ),
        hovertemplate="Ошибка: %{x:.1f}%<br>Плотность: %{y:.4f}<extra></extra>"
    ))

    # KDE-кривая плотности
    if len(error_pct_display) >= 30:
        kde_x, kde_y = _compute_kde(error_pct_display, n_points=200)
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode="lines",
            name="Плотность (KDE)",
            line=dict(color=COLORS["kde_line"], width=2.5),
            hovertemplate="Ошибка: %{x:.1f}%<br>Плотность: %{y:.4f}<extra></extra>"
        ))

    reference_lines = [
        {
            "x": 0,
            "label": "Нулевая ошибка",
            "color": COLORS["zero_line"],
            "dash": "dash"
        },
        {
            "x": mean_err,
            "label": f"Среднее: {mean_err:.1f}%",
            "color": COLORS["mean_line"],
            "dash": "dot"
        },
        {
            "x": median_err,
            "label": f"Медиана: {median_err:.1f}%",
            "color": COLORS["median_line"],
            "dash": "dot"
        },
    ]

    _add_reference_lines(fig, reference_lines, error_pct_display)

    # Адаптивный bargap: при малом числе бинов - шире зазоры для читаемости
    bargap = 0.08 if n_bins <= 15 else 0.04 if n_bins <= 30 else 0.02

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
            title="Плотность вероятности",
            gridcolor=COLORS["grid"],
            showgrid=True
        ),
        height=500,
        bargap=bargap,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.75,
            bgcolor="rgba(255, 255, 255, 0.8)",
            font=dict(size=11)
        ),
        margin=dict(l=60, r=160, t=80, b=60)
    )

    # Аннотация со сводной статистикой
    clipped_note = f"<br>• Обрезано: {n_clipped:,} выбросов" if n_clipped > 0 else ""

    stats_text = (
        f"<b>Статистика ошибок:</b><br>"
        f"• Среднее: {mean_err:.1f} %<br>"
        f"• Медиана: {median_err:.1f} %<br>"
        f"• Стд. откл.: {std_err:.1f} %<br>"
        f"• Мин / Макс: {min_err:.1f} % / {max_err:.1f} %<br>"
        f"• Наблюдений: {n_total:,} <br>"
        f"• Бинов: {n_bins} <br>"
        f"{clipped_note}"
    )

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

def _compute_adaptive_bins(data: np.ndarray) -> int:
    """Sturges/Freedman-Diaconis, клэмп 20-80 бинов"""
    n = len(data)

    sturges = int(np.ceil(1 + np.log2(max(n, 1))))

    if n < 50:
        n_bins = sturges
    else:
        # Freedman-Diaconis
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        if iqr < 1e-9:
            # Вырожденный случай: все значения почти одинаковы
            n_bins = 10
        else:
            bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
            data_range = float(np.max(data) - np.min(data))
            fd_bins = int(np.ceil(data_range / bin_width))

    return max(fd_bins, sturges)

def _compute_kde(data: np.ndarray, n_points: int = 200) -> tuple:
    """Ручной KDE через гауссово ядро - без scipy.stats зависимости"""
    KDE_SAMPLE_LIMIT = 10_000

    n = len(data)
    rng = np.random.default_rng(seed=get_model_config().get("random_state", 42))

    # Подвыборка для больших массивов (сохраняем воспроизводимость через seed)
    if n > KDE_SAMPLE_LIMIT:
        kde_data = rng.choice(data, size=KDE_SAMPLE_LIMIT, replace=False)
        n_effective = KDE_SAMPLE_LIMIT
    else:
        kde_data = data
        n_effective = n

    std = float(np.std(kde_data, ddof=1)) if n_effective > 1 else 1.0

    # Ширина ядра по правилу Silverman
    q75, q25 = np.percentile(kde_data, [75, 25])
    iqr = q75 - q25

    # Защита от вырожденных случаев
    spread = min(std, iqr / 1.34) if iqr > 1e-9 else std
    spread = max(spread, 1e-6)
    bandwidth = 0.9 * spread * (n_effective ** (-0.2))

    # Сетка точек для KDE (с запасом 5% по краям для плавности)
    x_min = float(np.min(kde_data))
    x_max = float(np.max(kde_data))
    margin = (x_max - x_min) * 0.05
    x_grid = np.linspace(x_min - margin, x_max + margin, n_points)

    # Векторизованное гауссово ядро через broadcasting
    u = (x_grid[:, np.newaxis] - kde_data[np.newaxis, :]) / bandwidth
    density = np.mean(np.exp(-0.5 * u * u), axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))

    return x_grid, density

def create_store_comparison(full_data: pd.DataFrame, selected_store: int, start_date: pd.Timestamp, end_date: pd.Timestamp) -> go.Figure:
    """Grouped bar: avg actual vs predicted по магазинам + mape каждого"""
    if start_date is None or end_date is None:
        return create_empty_chart("Выберите диапазон дат для сравнения")

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except Exception as e:
        logger.error(f"Ошибка парсинга дат для сравнения магазинов: {e}")
        return create_empty_chart(f"Ошибка в формате дат: {e}")

    # Фильтрация по датам
    date_mask = (full_data["Date"] >= start_date) & (full_data["Date"] <= end_date)
    filtered_data = full_data.loc[date_mask]

    if len(filtered_data) == 0:
        return create_empty_chart("Нет данных для сравнения")

    # Векторизованная агрегация
    grouped = filtered_data.groupby("Store").agg(
        AvgSales=("ActualSales", "mean"),
        StdSales=("ActualSales", "std"),
        TotalSales=("ActualSales", "sum"),
        AvgPredicted=("PredictedSales", "mean")
    ).round(2)

    # Векторизованный расчет mape по магазинам
    filtered_data = filtered_data.copy()
    filtered_data["APE"] = np.where(
        filtered_data["ActualSales"] > 0,
        np.abs(filtered_data["ActualSales"] - filtered_data["PredictedSales"]) / filtered_data["ActualSales"] * 100,
        np.nan
    )
    store_mape = filtered_data.groupby("Store")["APE"].mean()
    grouped["MAPE"] = store_mape.reindex(grouped.index).fillna(0.0)

    # Топ-15 магазинов по средним продажам
    store_stats = grouped.sort_values("AvgSales", ascending=False).head(15)

    if len(store_stats) == 0:
        return create_empty_chart("Нет данных для сравнения магазинов")

    fig = go.Figure()

    store_labels = store_stats.index.astype(str)

    # Столбцы - средние фактические продажи
    fig.add_trace(go.Bar(
        x=store_labels,
        y=store_stats["AvgSales"],
        name="Средние продажи (факт)",
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
        customdata=np.column_stack([store_stats["StdSales"], store_stats["MAPE"]])
    ))

    # Линия прогнозируемых продаж
    fig.add_trace(go.Bar(
        x=store_labels,
        y=store_stats["AvgPredicted"],
        name="Средние продажи (прогноз)",
        marker=dict(
            color=COLORS["predicted"],
            opacity=0.6,
            line=dict(color="rgba(231, 76, 60, 0.9)", width=0.5)
        ),
        hovertemplate="<b>Магазин %{x}</b><br>Прогноз: %{y:,.0f} €<extra></extra>"
    ))

    # Подсветка выбранного магазина
    if selected_store and selected_store in store_stats.index:
        selected_idx = list(store_stats.index).index(selected_store)
        fig.add_trace(go.Scatter(
            x=[store_labels[selected_idx]],
            y=[store_stats.loc[selected_store, "AvgSales"]],
            mode="markers",
            marker=dict(
                size=18,
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
            customdata=[store_stats.loc[selected_store, "MAPE"]]
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
        height=500,
        barmode="group",
        hovermode="x unified",
        legend=HORIZONTAL_LEGEND
    )

    return fig

def create_feature_importance_chart() -> go.Figure:
    """Горизонтальный barh с viridis-градиентом, top N признаков"""
    top_n = get_reporting_config().get("feature_importance_top_n", 20)
    importance_path = REPORTS_PATH / f"feature_importance_top{top_n}.csv"

    if os.path.exists(importance_path):
        try:
            # Проверяем кэш по времени модификации файла
            mtime = os.path.getmtime(importance_path)
            return _build_feature_importance_fig(str(importance_path), mtime)

        except Exception as e:
            logger.error(f"Ошибка загрузки важности признаков: {e}")

    # Если файл не найден или ошибка - создаем информационный график
    logger.warning(f"Файл важности признаков не найден: {importance_path}. График будет доступен после обучения модели")
    return create_info_chart(
        "Важность признаков модели",
        "График будет доступен после обучения модели.<br>"
        "Файл lgbm_feature_importance.csv не найден."
    )

@lru_cache(maxsize=4)
def _build_feature_importance_fig(path: str, mtime: float) -> go.Figure:
    """Обертка с @lru_cache - модель не меняется между callback"""
    importance_df = pd.read_csv(path)

    # Сортировка и выбор топ-20
    importance_df = importance_df.sort_values("importance", ascending=True).tail(20)

    fig = go.Figure()

    # Нормализация для цветовой шкалы
    imp_min = importance_df["importance"].min()
    imp_max = importance_df["importance"].max()
    imp_range = max(imp_max - imp_min, 1e-9)
    norm_importance = (importance_df["importance"] - imp_min) / imp_range

    bar_colors = [VIRIDIS_PALETTE[int(v * (len(VIRIDIS_PALETTE) - 1))] for v in norm_importance]

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
            title="Важность признаков",
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

    logger.info(f"График важности признаков построен: {len(importance_df)} признаков из {path}")

    return fig

def create_sales_trend_chart(data: pd.DataFrame) -> go.Figure:
    """Дневные продажи + rolling mean для сглаживания шума"""
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
        legend=HORIZONTAL_LEGEND
    )

    return fig


def create_data_table(data: pd.DataFrame) -> go.Figure:
    """Plotly Table: Store, Date, Actual, Predicted, Error, mape%"""
    if len(data) == 0:
        return create_empty_chart("Нет данных для таблицы")

    # Выбираем нужные колонки и сортируем
    cols = ["Store", "Date", "ActualSales", "PredictedSales"]
    table_data = data[cols].copy()

    # Добавляем расчетные колонки (векторизованно)
    table_data["Error"] = table_data["ActualSales"] - table_data["PredictedSales"]
    table_data["ErrorPct"] = np.where(
        table_data["ActualSales"] > 0,
        (table_data["Error"] / table_data["ActualSales"] * 100).round(2),
        0.0
    )

    # Сортировка по дате и магазину
    table_data = table_data.sort_values(["Date", "Store"])

    # Ограничиваем количество строк для производительности
    total_rows = len(table_data)
    table_data = table_data.head(1000)

    # Цвет ячеек ошибки: зеленый - недопрогноз, красный - перепрогноз
    error_colors = [
        "rgba(39, 174, 96, 0.1)" if val >= 0 else "rgba(231, 76, 60, 0.1)"
        for val in table_data["Error"]
    ]

    # Создаем таблицу
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[
                "<b>Магазин</b>", "<b>Дата</b>", "<b>Факт (€)</b>",
                "<b>Прогноз (€)</b>", "<b>Ошибка (€)</b>", "<b>Ошибка (%)</b>"
            ],
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
    """Заглушка с иконкой и текстом, когда данных нет"""
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
    """Пустой figure c текстом по центру"""
    return create_info_chart("Нет данных", message)
