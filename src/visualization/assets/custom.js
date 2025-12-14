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
    }

    function setupMetricAnimations() {
        function animateMetrics() {
            const metrics = document.querySelectorAll(".metric-value");
            metrics.forEach(metric => {
                const originalText = metric.textContent;

                // Проверка, что значение можно анимировать
                if (!originalText || originalText === "0" || originalText === "-" || originalText.includes("Нет данных")) {
                    return;
                }

                metric.textContent = "0";

                let counter = 0;
                const cleanText = originalText.replace(/[^0-9.-]+/g, "");
                const targetValue = parseFloat(cleanText) || 0;

                if (targetValue === 0) return;

                const increment = targetValue / 50;
                const isCurrency = originalText.includes("€");
                const isPercentage = originalText.includes("%");

                const timer = setInterval(() => {
                    counter += increment;
                    if (counter >= targetValue) {
                        counter = targetValue;
                        clearInterval(timer);
                    }

                    if (isCurrency) {
                        metric.textContent = "€" + Math.floor(counter).toLocaleString();
                    } else if (isPercentage) {
                        metric.textContent = counter.toFixed(1) + "%";
                    } else {
                        metric.textContent = Math.floor(counter).toLocaleString();
                    }
                }, 30);
            });
        }

        // Запуск анимации при загрузке
        setTimeout(animateMetrics, 1000);

        // Обновление анимации при обновлении данных
        const refreshButton = document.getElementById("refresh-btn");
        if (refreshButton) {
            refreshButton.addEventListener("click", function () {
                setTimeout(animateMetrics, 1500);
            });
        }
    }

    function setupControlHighlighting() {
        const controls = document.querySelectorAll(".control-group select, .control-group input, .control-group button");
        controls.forEach(control => {
            control.addEventListener("focus", function (){
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

    function setupAutoRefresh() {
        let autoRefreshInterval = null;
        let lastAutoRefresh = 0;

        function startAutoRefresh() {
            // Очищаем существующий интервал
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }

            // Запускаем новый интервал
            autoRefreshInterval = setInterval(() => {
                const now = Date.now();

                // Не обновляем чаще чем раз в 10 секунд
                if (now - lastAutoRefresh < 10000) {
                    return;
                }

                const refreshBtn = document.getElementById("refresh-btn");
                if (refreshBtn && !refreshBtn.disabled) {
                    lastAutoRefresh = now;

                    // Безопасный клик с проверкой
                    try {
                        refreshBtn.click();
                        console.log("Автообновление: " + new Date().toLocaleString());
                    } catch (error) {
                        console.error("Ошибка автообновления:", error);
                    }
                }
            }, 300000);  // 5 минут
        }

        // Очистка при уходе со страницы
        window.addEventListener('beforeunload', function() {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
        });

        // Очистка при скрытии страницы
        document.addEventListener('visibilitychange', function() {
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

    function setupErrorHandling() {
        // Глобальный обработчик ошибок JavaScript
        window.addEventListener("error", function(e) {
            console.error("Ошибка дашборда:", e.error);
            showErrorMessage("Произошла ошибка в работе дашборда");
        });

        // Обработчик для неудачных промисов
        window.addEventListener("unhandledrejection", function(e) {
            console.error("Необработанная ошибка промиса:", e.reason);
            showErrorMessage("Ошибка при обновлении данных");
        });
    }

    function showErrorMessage(message) {
        // Создаем элемент для отображения ошибки
        let errorDiv = document.getElementById("dashboard-error");
        if (!errorDiv) {
            errorDiv = document.createElement("div");
            errorDiv.id = "dashboard-error";
            errorDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 5px;
                z-index: 1000;
                display: none;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
            `;
            document.body.appendChild(errorDiv);
        }

        errorDiv.textContent = message;
        errorDiv.style.display = "block";

        // Автоматическое скрытие через 5 секунд
        setTimeout(() => {
            errorDiv.style.display = "none";
        }, 5000);

        // Добавляем кнопку закрытия
        if (!errorDiv.querySelector('.error-close-btn')) {
            const closeBtn = document.createElement("button");
            closeBtn.textContent = "×";
            closeBtn.className = "error-close-btn";
            closeBtn.style.cssText = `
                position: absolute;
                top: 5px;
                right: 10px;
                background: none;
                border: none;
                color: white;
                font-size: 20px;
                cursor: pointer;
                padding: 0;
                line-height: 1;
            `;
            closeBtn.onclick = () => {
                errorDiv.style.display = "none";
            };
            errorDiv.appendChild(closeBtn);
        }
    }

    // Дополнительная функция для плавного скролла к графику при выборе
    function setupSmoothScrolling() {
        const storeSelector = document.getElementById("store-selector");
        if (storeSelector) {
            storeSelector.addEventListener("change", function() {
                // Плавный скролл к первому графику после изменения магазина
                setTimeout(() => {
                    const firstChart = document.querySelector(".chart-container");
                    if (firstChart) {
                        firstChart.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }
                }, 300);
            });
        }
    }

    // Инициализация плавного скролла
    setupSmoothScrolling();
});