import numpy as np
from scipy import stats
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from config.settings import REPORTS_PATH
import json

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-whitegrid")

class BaseModels:
    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.model: Optional[Any] = None

    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series, List[float]], y_pred: Union[np.ndarray, pd.Series, List[float]]) -> Dict[str, float]:
        """Расчет метрик качества"""
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        if len(y_true_array) != len(y_pred_array):
            raise ValueError("Длины y_true и y_pred не совпадают")

        mae = mean_absolute_error(y_true_array, y_pred_array)
        rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
        mape = np.mean(np.abs((y_true_array - y_pred_array) / np.maximum(y_true_array, 1))) * 100
        r2 = r2_score(y_true_array, y_pred_array)
        max_err = np.max(np.abs(y_true_array - y_pred_array))

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "R2": float(r2),
            "MaxError": float(max_err),
            "Samples": len(y_true)
        }

    def plot_predictions(self, y_true: Union[np.ndarray, pd.Series, List[float]], y_pred: Union[np.ndarray, pd.Series, List[float]], title: str, store_id: Optional[int] = None) -> None:
        """Визуализация прогнозов"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Цветовая схема
        colors = ["#2E86AB", "#A23B72"]

        # Основной график - прогнозы vs факт
        sample_size = min(100, len(y_true))
        indices = range(sample_size)

        # Преобразование в numpy arrays для гарантии корректного доступа
        y_true_values = np.array(y_true)
        y_pred_values = np.array(y_pred)

        ax1.plot(indices, y_true_values[:sample_size],
                 label="Фактические продажи", color=colors[0], linewidth=2.5, marker="o", markersize=4)

        ax1.plot(indices, y_pred_values[:sample_size],
                 label="Прогноз модели", color=colors[1], linewidth=2, linestyle="--", marker="s", markersize=3)

        # Расчет метрик
        metrics = self.calculate_metrics(y_true, y_pred)

        # Заголовок и оформление
        store_info = f" - Магазин {store_id}" if store_id else ""
        ax1.set_title(f"{title}{store_info}\n", fontsize=16, fontweight="bold", pad=20)

        # Вывод метрик на графике
        metric_text = f"MAPE: {metrics["MAPE"]:.1f}% | R²: {metrics["R2"]:.3f} | RMSE: {metrics["RMSE"]:.0f} €"
        ax1.text(0.02, 0.98, metric_text, transform=ax1.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax1.set_ylabel("Продажи, €", fontsize=12)
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # График ошибок
        errors = y_true_values[:sample_size] - y_pred_values[:sample_size]
        ax2.bar(indices, errors, alpha=0.7, color="#F18F01", label="Ошибка прогноза")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8)
        ax2.set_xlabel("Временные точки", fontsize=12)
        ax2.set_ylabel("Ошибка, €", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохранение графика
        filename = f"{title.replace(" ", "_")}_{store_id if store_id else "all"}.png"
        plt.savefig(REPORTS_PATH / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_residuals(self, y_true: Union[np.ndarray, pd.Series, List[float]], y_pred: Union[np.ndarray, pd.Series, List[float]], title: str) -> None:
        """Анализ остатков"""
        residuals = np.array(y_true) - np.array(y_pred)
        metrics = self.calculate_metrics(y_true, y_pred)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Анализ остатков модели - {title}\nMAPE: {metrics["MAPE"]:.2f}%",
                     fontsize=16, fontweight="bold", y=0.95)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

        # 1. Остатки vs Прогноз
        axes[0,0].scatter(y_pred, residuals, alpha=0.6, color=colors[0], s=50)
        axes[0,0].axhline(y=0, color=colors[1], linestyle="--", linewidth=2, alpha=0.8)
        axes[0,0].set_xlabel("Прогнозируемые продажи, €", fontsize=11)
        axes[0,0].set_ylabel("Остатки, €", fontsize=11)
        axes[0,0].set_title("Остатки vs Прогноз", fontsize=13, fontweight="bold")
        axes[0,0].grid(True, alpha=0.3)

        # Добавление линии тренда
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(y_pred, p(y_pred), color=colors[1], linewidth=2, alpha=0.8,
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
        autocorrelation_plot(residuals, ax=axes[1, 1], color=colors[0])
        axes[1, 1].set_title("Автокорреляция остатков", fontsize=13, fontweight="bold")
        axes[1, 1].set_ylim([-0.2, 0.2])
        axes[1, 1].grid(True, alpha=0.3)

        # Добавление доверительных интервалов
        axes[1, 1].axhline(y=-1.96 / np.sqrt(len(residuals)), color=colors[1], linestyle="--", alpha=0.7)
        axes[1, 1].axhline(y=1.96 / np.sqrt(len(residuals)), color=colors[1], linestyle="--", alpha=0.7)

        plt.tight_layout()

        filename = f"residuals_analysis_{title.replace(" ", "_")}.png"
        plt.savefig(REPORTS_PATH / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, top_n: int = 20) -> None:
        """Визуализация важности признаков"""
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_scores
        }).sort_values("importance", ascending=True).tail(top_n)

        # Создание градиентной цветовой схемы
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))

        fig, ax = plt.subplots(figsize=(12, 10))

        bars = ax.barh(importance_df["feature"], importance_df["importance"],
                       color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Добавление значений на барчарт
        for i, (bar, value) in enumerate(zip(bars, importance_df["importance"])):
            ax.text(bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Важность признака", fontsize=12, fontweight="bold")
        ax.set_ylabel("Признаки", fontsize=12, fontweight="bold")
        ax.set_title(f"Топ-{top_n} самых важных признаков модели",
                     fontsize=14, fontweight="bold", pad=20)

        # Улучшение читаемости подписей
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        # Добавление информационной панели
        total_importance = importance_df["importance"].sum()
        explained_text = f"Общая важность: {total_importance:.3f}\n"
        explained_text += f"Лучший признак: {importance_df["feature"].iloc[-1]}"

        ax.text(0.02, 0.98, explained_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        filename = f"feature_importance_top{top_n}.png"
        plt.savefig(REPORTS_PATH / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]], title: str ="Сравнение моделей") -> None:
        """Сравнение метрик нескольких моделей"""
        models = list(metrics_dict.keys())
        metrics_to_plot = ["MAPE", "R2", "RMSE"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for idx, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[model][metric] for model in models]

            # Для MAPE и RMSE - чем меньше, тем лучше (красный для худших)
            if metric in ["MAPE", "RMSE"]:
                bar_colors = [colors[1] if value == max(values) else colors[0] for value in values]
            else:  # Для R2 - чем больше, тем лучше (зеленый для лучших)
                bar_colors = [colors[2] if value == max(values) else colors[0] for value in values]

            bars = axes[idx].bar(models, values, color=bar_colors, alpha=0.8, edgecolor="black")

            # Добавление значений на столбцы
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
        plt.savefig(REPORTS_PATH / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "model_metrics.json") -> None:
        """Сохранение метрик в файл"""
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value

        with open(REPORTS_PATH / filename, "w", encoding="utf-8") as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)