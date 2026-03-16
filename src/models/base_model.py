import json
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf

from config.settings import REPORTS_PATH, get_dashboard_config, get_model_config, get_reporting_config

plt.style.use(get_dashboard_config().get("plot_style", "seaborn-v0_8-whitegrid"))

class BaseModels:
    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.model: Optional[Any] = None
        self.logger = logging.getLogger(self.__class__.__name__)

        plot_style = get_dashboard_config().get("plot_style", "seaborn-v0_8-whitegrid")
        plt.style.use(plot_style)

    def _savefig(self, filepath) -> None:
        """Общий savefig с dpi/bbox из конфига"""
        report_cfg = get_reporting_config()
        plt.savefig(
            filepath,
            dpi=report_cfg.get("plot_dpi", 300),
            bbox_inches=report_cfg.get("plot_bbox", "tight"),
            facecolor=report_cfg.get("plot_facecolor", "white")
        )

    def calculate_metrics(self, y_true: np.ndarray | pd.Series | list[float], y_pred: np.ndarray | pd.Series | list[float]) -> dict[str, float]:
        """mae, rmse, mape, rmspe, r2, medae, bias, maxerror"""
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        if len(y_true_array) != len(y_pred_array):
            raise ValueError("Длины y_true и y_pred не совпадают")

        mae = mean_absolute_error(y_true_array, y_pred_array)
        rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
        mape = np.mean(np.abs((y_true_array - y_pred_array) / np.maximum(y_true_array, 1))) * 100
        r2 = r2_score(y_true_array, y_pred_array)
        max_err = np.max(np.abs(y_true_array - y_pred_array))

        # rmspe штрафует большие относительные ошибки сильнее чем mape
        rmspe = np.sqrt(np.mean(((y_true_array - y_pred_array) / np.maximum(y_true_array, 1)) ** 2)) * 100

        med_ae = np.median(np.abs(y_true_array - y_pred_array))
        bias = np.mean(y_true_array - y_pred_array)

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "RMSPE": float(rmspe),
            "R2": float(r2),
            "MaxError": float(max_err),
            "MedAE": float(med_ae),
            "Bias": float(bias),
            "Samples": len(y_true)
        }

    def _get_mape_color(self, mape: float) -> str:
        """зеленый/оранжевый/красный по порогам из конфига"""
        thresholds = get_model_config().get("quality_thresholds", {})
        excellent = thresholds.get("excellent", 10)
        good = thresholds.get("good", 15)

        if mape < excellent:
            return "#27AE60" # Зелёный - отличное качество
        if mape < good:
            return "#F39C12" # Оранжевый - хорошее качество
        return "#E74C3C" # Красный - требует улучшения

    def plot_predictions(self, y_true: np.ndarray | pd.Series | list[float], y_pred: np.ndarray | pd.Series | list[float], title: str, store_id: int | None = None, sample_size: int = 100) -> None:
        """Факт vs прогноз + график ошибок внизу"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        colors = ["#2E86AB", "#A23B72"]

        sample_size = min(sample_size, len(y_true))
        indices = range(sample_size)

        y_true_values = np.array(y_true)
        y_pred_values = np.array(y_pred)

        ax1.plot(indices, y_true_values[:sample_size],
                 label="Фактические продажи", color=colors[0], linewidth=2.5, marker="o", markersize=4)

        ax1.plot(indices, y_pred_values[:sample_size],
                 label="Прогноз модели", color=colors[1], linewidth=2, linestyle="--", marker="s", markersize=3)

        metrics = self.calculate_metrics(y_true, y_pred)

        store_info = f" - Магазин {store_id}" if store_id else ""
        ax1.set_title(f"{title}{store_info}\n", fontsize=16, fontweight="bold", pad=20)

        mape_color = self._get_mape_color(metrics["MAPE"])
        metric_text = f"MAPE: {metrics["MAPE"]:.1f}% | R²: {metrics["R2"]:.3f} | RMSE: {metrics["RMSE"]:.0f} €"
        ax1.text(0.02, 0.98, metric_text, transform=ax1.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor=mape_color, alpha=0.3))

        ax1.set_ylabel("Продажи, €", fontsize=12)
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        errors = y_true_values[:sample_size] - y_pred_values[:sample_size]
        bar_colors = ["#27AE60" if e >= 0 else "#E74C3C" for e in errors]
        ax2.bar(indices, errors, alpha=0.7, color=bar_colors)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8)

        mean_error = np.mean(errors)
        ax2.axhline(y=mean_error, color="#F39C12", linestyle="--", linewidth=1.5, label=f"Средняя ошибка: {mean_error:,.0f} €")
        ax2.legend(fontsize=10, loc="upper right")

        ax2.set_xlabel("Временные точки", fontsize=12)
        ax2.set_ylabel("Ошибка, €", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"{title.replace(" ", "_")}_{store_id if store_id else "all"}.png"
        self._savefig(REPORTS_PATH / filename)
        plt.close()

    def plot_residuals(self, y_true: np.ndarray | pd.Series | list[float], y_pred: np.ndarray | pd.Series | list[float], title: str, store_ids: np.ndarray | None = None) -> None:
        """6 графиков: scatter, histogram, Q-Q, ACF, CDF, boxplot по диапазонам"""
        residuals = np.array(y_true) - np.array(y_pred)
        metrics = self.calculate_metrics(y_true, y_pred)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Анализ остатков модели - {title}\nMAPE: {metrics["MAPE"]:.2f}%",
                     fontsize=16, fontweight="bold", y=0.98)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

        # 1. Остатки vs Прогноз
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=colors[0], s=50)
        axes[0, 0].axhline(y=0, color=colors[1], linestyle="--", linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel("Прогнозируемые продажи, €", fontsize=11)
        axes[0, 0].set_ylabel("Остатки, €", fontsize=11)
        axes[0, 0].set_title("Остатки vs Прогноз", fontsize=13, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)

        # Линия тренда
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        y_pred_sorted = np.sort(y_pred)
        axes[0, 0].plot(y_pred_sorted, p(y_pred_sorted), color=colors[1], linewidth=2, alpha=0.8,
                        label=f"Тренд (наклон: {z[0]:.3f})")
        axes[0, 0].legend()

        # 2. Распределение остатков
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=colors[2],
                        edgecolor="black", linewidth=0.5)
        axes[0, 1].axvline(x=0, color=colors[1], linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Остатки, €", fontsize=11)
        axes[0, 1].set_ylabel("Частота", fontsize=11)
        axes[0, 1].set_title(f"Распределение остатков\nμ: {np.mean(residuals):.1f}, σ: {np.std(residuals):.1f}",
                             fontsize=13, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Нормальный Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].get_lines()[0].set_marker("o")
        axes[1, 0].get_lines()[0].set_markersize(4)
        axes[1, 0].get_lines()[0].set_alpha(0.6)
        axes[1, 0].get_lines()[1].set_color(colors[3])
        axes[1, 0].get_lines()[1].set_linewidth(2)
        axes[1, 0].set_title("Q-Q Plot (проверка нормальности)", fontsize=13, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Автокорреляция остатков
        self._plot_autocorrelation(axes[1, 1], residuals, store_ids, colors)

        # 5. Кумулятивное распределение ошибок
        abs_errors = np.abs(residuals)
        sorted_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        axes[0, 2].plot(sorted_errors, cumulative, color=colors[0], linewidth=2)
        axes[0, 2].fill_between(sorted_errors, 0, cumulative, alpha=0.2, color=colors[0])

        # Отметки процентилей
        for pct in [50, 75, 95]:
            pct_value = np.percentile(abs_errors, pct)
            axes[0, 2].axvline(x=pct_value, color=colors[4], linestyle="--", alpha=0.7)
            axes[0, 2].axhline(y=pct, color=colors[3], linestyle=":", alpha=0.5)
            axes[0, 2].text(pct_value * 1.02, pct - 3, f"P{pct}: {pct_value:,.0f} €", fontweight="bold")

        axes[0, 2].set_xlabel("Абсолютная ошибка, €", fontsize=11)
        axes[0, 2].set_ylabel("Кумулятивный %", fontsize=11)
        axes[0, 2].set_title("Кумулятивное распределение ошибок", fontsize=13, fontweight="bold")
        axes[0, 2].set_ylim(0, 105)
        axes[0, 2].grid(True, alpha=0.3)

        # 6. Ошибки по диапазонам продаж (Box plot)
        y_pred_array = np.array(y_pred)
        n_bins = 5
        bin_edges = np.percentile(y_pred_array, np.linspace(0, 100, n_bins + 1))
        bin_labels = [f"{bin_edges[i]:,.0f}-{bin_edges[i+1]:,.0f}" for i in range(n_bins)]

        # Группировка остатков по диапазонам
        bin_indices = np.digitize(y_pred_array, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        grouped_residuals = [residuals[bin_indices == i] for i in range(n_bins)]

        bp = axes[1, 2].boxplot(grouped_residuals, labels=bin_labels, patch_artist=True)

        # Стилизация boxplot
        box_colors = [colors[0], colors[2], colors[4], colors[3], colors[1]]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[1, 2].axhline(y=0, color=colors[1], linestyle="--", linewidth=2)
        axes[1, 2].set_xlabel("Диапазон прогнозных продаж, €", fontsize=11)
        axes[1, 2].set_ylabel("Остатки, €", fontsize=11)
        axes[1, 2].set_title("Распределение ошибок по диапазонам", fontsize=13, fontweight="bold")
        axes[1, 2].tick_params(axis="x", rotation=15)
        axes[1, 2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename = f"residuals_analysis_{title.replace(" ", "_")}.png"
        self._savefig(REPORTS_PATH / filename)
        plt.close()

    def _plot_autocorrelation(self, ax, residuals: np.ndarray, store_ids: np.ndarray | None, colors: list[str]) -> None:
        """ACF по магазинам (усреднение) - ищем недельный/месячный паттерн"""
        max_lags = 60 # Дней

        if store_ids is not None:
            unique_stores = np.unique(store_ids)
            acf_per_store = []

            for store_id in unique_stores:
                store_mask = store_ids == store_id
                store_residuals = residuals[store_mask]

                if len(store_residuals) > max_lags + 1:
                    try:
                        store_acf_values = acf(
                            store_residuals, nlags=max_lags, fft=True
                        )
                        acf_per_store.append(store_acf_values)
                    except Exception:
                        continue

            if len(acf_per_store) > 0:
                # Средняя ACF по всем магазинам
                mean_acf_values = np.mean(acf_per_store, axis=0)
                n_stores_used = len(acf_per_store)

                # Средний размер выборки магазина для доверительного интервала
                avg_store_length = np.mean([
                    np.sum(store_ids == sid) for sid in unique_stores
                ])
            else:
                # Fallback: если у всех магазинов мало данных
                self.logger.warning("Недостаточно данных для помогазинной ACF, используется общий срез")
                mean_acf_values = acf(
                    residuals[:min(len(residuals), 5000)],
                    nlags=max_lags, fft=True
                )
                n_stores_used = 0
                avg_store_length = len(residuals)
        else:
            # Без store_ids - считаем ACF по всем данным разом
            sample = residuals[:min(len(residuals), 5000)]
            mean_acf_values = acf(sample, nlags=max_lags, fft=True)
            n_stores_used = 0
            avg_store_length = len(sample)

        lags = np.arange(len(mean_acf_values))

        ax.bar(lags, mean_acf_values, width=0.8, color=colors[0], alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8)

        ci = 1.96 / np.sqrt(avg_store_length)
        ax.axhline(y=ci, color=colors[1], linestyle="--", alpha=0.7, label=f"95% ДИ (\u00b1{ci:.3f})")
        ax.axhline(y=-ci, color=colors[1], linestyle="--", alpha=0.7)

        ax.fill_between(lags, -ci, ci, alpha=0.1, color=colors[1])

        # Помечаем недельный (7д), двухнедельный (14д) и месячный (28д) лаги
        for key_lag, label in [(7, "7д"), (14, "14д"), (28, "28д")]:
            if key_lag < len(mean_acf_values):
                acf_at_lag = mean_acf_values[key_lag]
                marker_color = colors[1] if abs(acf_at_lag) > ci else colors[2]
                ax.plot(key_lag, acf_at_lag, "v", color=marker_color, markersize=8, zorder=5)
                ax.annotate(f"{label}\n{acf_at_lag:.3f}", xy=(key_lag, acf_at_lag), xytext=(key_lag + 2, acf_at_lag + 0.03), fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        if store_ids is not None and n_stores_used > 0:
            ax.set_title(f"Автокорреляция остатков\n(средняя по {n_stores_used} магазинам)", fontsize=13, fontweight="bold")
        else:
            ax.set_title("Автокорреляция остатков", fontsize=13, fontweight="bold")

        ax.set_xlabel("Лаг (дни)", fontsize=11)
        ax.set_ylabel("ACF", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    def plot_feature_importance(self, feature_names: list[str], importance_scores: np.ndarray | None = None, top_n: int | None = None) -> None:
        """Горизонтальный barh - top N признаков по importance"""
        if top_n is None:
            top_n = get_reporting_config().get("feature_importance_top_n", 20)

        # Получение importance из модели, если не передано
        if importance_scores is None:
            if self.model is None or not hasattr(self.model, "feature_importances_"):
                self.logger.warning("Модель не обучена или не поддерживает feature_importances_")
                return
            importance_scores = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_scores
        }).sort_values("importance", ascending=True).tail(top_n)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))

        fig, ax = plt.subplots(figsize=(12, 10))

        bars = ax.barh(importance_df["feature"], importance_df["importance"],
                       color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        for bar, value in zip(bars, importance_df["importance"]):
            ax.text(bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Важность признака", fontsize=12, fontweight="bold")
        ax.set_ylabel("Признаки", fontsize=12, fontweight="bold")
        ax.set_title(f"Топ-{top_n} самых важных признаков модели",
                     fontsize=14, fontweight="bold", pad=20)

        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        total_importance = importance_df["importance"].sum()
        explained_text = f"Общая важность: {total_importance:.3f}\n"
        explained_text += f"Лучший признак: {importance_df["feature"].iloc[-1]}"

        ax.text(0.02, 0.98, explained_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        filename = f"feature_importance_top{top_n}.png"
        self._savefig(REPORTS_PATH / filename)
        plt.close()

        importance_data_path = REPORTS_PATH / f"feature_importance_top{top_n}.csv"
        importance_df.to_csv(importance_data_path, index=False)

        self.logger.info(f"График важности признаков сохранен: {REPORTS_PATH / filename}")
        self.logger.info(f"Данные важности признаков сохранены: {importance_data_path}")

    def plot_metrics_comparison(self, metrics_dict: dict[str, dict[str, float]], title: str ="Сравнение моделей") -> None:
        """4 барчарта рядом: mape, r2, rmse, rmspe"""
        models = list(metrics_dict.keys())
        metrics_to_plot = ["MAPE", "R2", "RMSE", "RMSPE"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for idx, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[model][metric] for model in models]

            # Красный = худший, зеленый = лучший
            if metric in ["MAPE", "RMSPE", "RMSE"]:
                bar_colors = [colors[1] if value == max(values) else colors[0] for value in values]
            else:  # Для r2 - чем больше, тем лучше (зеленый для лучших)
                bar_colors = [colors[2] if value == max(values) else colors[0] for value in values]

            bars = axes[idx].bar(models, values, color=bar_colors, alpha=0.8, edgecolor="black")

            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                               f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

            axes[idx].set_title(f"{metric}", fontsize=13, fontweight="bold")
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].tick_params(axis="x", rotation=45)
            axes[idx].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        filename = "models_comparison.png"
        self._savefig(REPORTS_PATH / filename)
        plt.close()

    @staticmethod
    def _to_serializable(data: dict) -> dict:
        """numpy типы -> python native для json.dump"""
        result = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                result[key] = int(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
            elif isinstance(value, np.bool_):
                result[key] = bool(value)
            else:
                result[key] = value
        return result

    def save_metrics(self, metrics: dict[str, Any], filename: str = "model_metrics.json") -> None:
        """Метрики -> json, numpy типы -> python native"""

        with open(REPORTS_PATH / filename, "w", encoding="utf-8") as f:
            json.dump(self._to_serializable(metrics), f, indent=2, ensure_ascii=False)
