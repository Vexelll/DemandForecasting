document.addEventListener("DOMContentLoaded", function () {
  console.log("Demand Forecasting Dashboard custom JS loaded");

  // Инициализация
  initDashboard();

  function initDashboard() {
    // Анимации загрузки метрик
    setupMetricAnimations();

    // Подсветка активного элемента управления
    setupControlHighlighting();

    // Автоматическое обновление данных
    setupAutoRefresh();

    // Обработка ошибок
    setupErrorHandling();

    // Плавный скролл к графикам
    setupSmoothScrolling();
  }

  // Анимация метрик
  function setupMetricAnimations() {
    // MutationObserver для отслеживания обновления метрик через Dash callbacks
    const metricsContainer = document.querySelector(".metrics-grid");
    if (!metricsContainer) return;

    const observer = new MutationObserver(function () {
      // Небольшая задержка, чтобы Dash успел обновить DOM
      setTimeout(animateMetrics, 100);
    });

    observer.observe(metricsContainer, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    // Первоначальная анимация при загрузке
    setTimeout(animateMetrics, 1000);
  }

  function animateMetrics() {
    const metrics = document.querySelectorAll(".metric-value");
    metrics.forEach(function (metric) {
      var originalText = metric.textContent;

      // Проверка: можно ли анимировать значение
      if (
        !originalText ||
        originalText === "0" ||
        originalText === "—" ||
        originalText === "-" ||
        originalText.includes("Нет данных")
      ) {
        return;
      }

      // Предотвращение повторной анимации (data-атрибут как флаг)
      if (metric.dataset.animating === "true") return;
      metric.dataset.animating = "true";

      var cleanText = originalText
        .replace(/[^0-9.,\-]+/g, "")
        .replace(/\s/g, "");
      var targetValue = parseFloat(cleanText) || 0;

      if (targetValue === 0) {
        metric.dataset.animating = "false";
        return;
      }

      metric.textContent = "0";

      var counter = 0;
      var increment = targetValue / 40;
      var isCurrency = originalText.includes("€");
      var isPercentage = originalText.includes("%");

      var timer = setInterval(function () {
        counter += increment;

        if (counter >= targetValue) {
          // Завершение: возвращаем оригинальный текст
          metric.textContent = originalText;
          metric.dataset.animating = "false";
          clearInterval(timer);
          return;
        }

        if (isCurrency) {
          metric.textContent =
            Math.floor(counter).toLocaleString("ru-RU") + " €";
        } else if (isPercentage) {
          metric.textContent = counter.toFixed(1) + "%";
        } else {
          metric.textContent = Math.floor(counter).toLocaleString("ru-RU");
        }
      }, 25);
    });
  }

  // Подсветка контролов при фокусе
  function setupControlHighlighting() {
    var controls = document.querySelectorAll(
      ".control-group select, .control-group input, .control-group button",
    );
    controls.forEach(function (control) {
      control.addEventListener("focus", function () {
        this.parentElement.style.backgroundColor = "#f8f9fa";
        this.parentElement.style.borderLeft = "3px solid #3498db";
        this.parentElement.style.transition = "all 0.3s ease";
      });

      control.addEventListener("blur", function () {
        this.parentElement.style.backgroundColor = "";
        this.parentElement.style.borderLeft = "";
      });
    });
  }

  // Автоматическое обновление (каждые 5 минут)
  function setupAutoRefresh() {
    var autoRefreshInterval = null;
    var lastAutoRefresh = 0;
    var REFRESH_INTERVAL_MS = 300000; // 5 минут
    var MIN_REFRESH_GAP_MS = 10000; // Минимум 10 секунд между обновлениями

    function startAutoRefresh() {
      // Очищаем существующий интервал
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }

      autoRefreshInterval = setInterval(function () {
        var now = Date.now();

        // Не обновляем чаще чем раз в MIN_REFRESH_GAP_MS
        if (now - lastAutoRefresh < MIN_REFRESH_GAP_MS) {
          return;
        }

        var refreshBtn = document.getElementById("refresh-btn");
        if (refreshBtn && !refreshBtn.disabled) {
          lastAutoRefresh = now;

          try {
            refreshBtn.click();
            console.log("Автообновление: " + new Date().toLocaleString());
          } catch (error) {
            console.error("Ошибка автообновления:", error);
          }
        }
      }, REFRESH_INTERVAL_MS);
    }

    // Запуск автообновления при загрузке
    startAutoRefresh();

    // Очистка при уходе со страницы
    window.addEventListener("beforeunload", function () {
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }
    });

    // Пауза при скрытии вкладки, возобновление при возврате
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        if (autoRefreshInterval) {
          clearInterval(autoRefreshInterval);
          autoRefreshInterval = null;
        }
      } else {
        startAutoRefresh();
      }
    });
  }

  // Глобальная обработка ошибок
  function setupErrorHandling() {
    window.addEventListener("error", function (e) {
      console.error("Ошибка дашборда:", e.error);
      showErrorMessage("Произошла ошибка в работе дашборда");
    });

    window.addEventListener("unhandledrejection", function (e) {
      console.error("Необработанная ошибка промиса:", e.reason);
      showErrorMessage("Ошибка при обновлении данных");
    });
  }

  function showErrorMessage(message) {
    var errorDiv = document.getElementById("dashboard-error");
    if (!errorDiv) {
      errorDiv = document.createElement("div");
      errorDiv.id = "dashboard-error";
      errorDiv.style.cssText =
        "position: fixed; top: 20px; right: 20px; " +
        "background: #e74c3c; color: white; padding: 15px 40px 15px 15px; " +
        "border-radius: 8px; z-index: 1000; display: none; " +
        "max-width: 320px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); " +
        "font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; " +
        "font-size: 14px; line-height: 1.4;";
      document.body.appendChild(errorDiv);

      // Кнопка закрытия
      var closeBtn = document.createElement("button");
      closeBtn.textContent = "×";
      closeBtn.style.cssText =
        "position: absolute; top: 8px; right: 12px; " +
        "background: none; border: none; color: white; " +
        "font-size: 20px; cursor: pointer; padding: 0; line-height: 1;";
      closeBtn.onclick = function () {
        errorDiv.style.display = "none";
      };
      errorDiv.appendChild(closeBtn);
    }

    // Устанавливаем текст (сохраняя кнопку закрытия)
    errorDiv.childNodes[0].textContent = message;
    errorDiv.style.display = "block";

    // Автоматическое скрытие через 5 секунд
    setTimeout(function () {
      errorDiv.style.display = "none";
    }, 5000);
  }

  // Плавный скролл к графикам
  function setupSmoothScrolling() {
    var storeSelector = document.getElementById("store-selector");
    if (storeSelector) {
      storeSelector.addEventListener("change", function () {
        setTimeout(function () {
          var firstChart = document.querySelector(".chart-container");
          if (firstChart) {
            firstChart.scrollIntoView({
              behavior: "smooth",
              block: "start",
            });
          }
        }, 300);
      });
    }
  }
});
