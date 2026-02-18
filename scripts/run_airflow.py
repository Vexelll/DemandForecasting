import logging
import subprocess
import sys
import time
import traceback
import shlex
import requests
from datetime import datetime
from pathlib import Path


class AirflowRunner:
    """Класс для управления локальным запуском Airflow через WSL"""

    # Базовые переменные окружения Airflow (единое место конфигурации)
    AIRFLOW_ENV_TEMPLATE = (
        "export AIRFLOW_HOME={airflow_home} && "
        "export AIRFLOW__CORE__DAGS_FOLDER={airflow_home}/dags && "
        "export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
        "export AIRFLOW__CORE__EXECUTOR=SequentialExecutor && "
        "export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="
        "'sqlite:///{airflow_home}/airflow.db' && "
        "export AIRFLOW__LOGGING__BASE_LOG_FOLDER={airflow_home}/logs"
    )

    # Максимальное время ожидания готовности API-сервера (секунды)
    API_HEALTH_TIMEOUT = 60
    API_HEALTH_INTERVAL = 3

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.airflow_home = self.project_root / "airflow"
        self.wsl_project_root = self._convert_to_wsl_path(self.project_root)
        self.wsl_airflow_home = self.wsl_project_root + "/airflow"
        self.logger = self._setup_logger()
        self._validate_paths()

    def _setup_logger(self):
        """Настройка системы логирования"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Консольный handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Форматтер
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Файловый handler
            log_file = self.project_root / "logs/airflow_runner.log"
            log_file.parent.mkdir(exist_ok=True, parents=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _validate_paths(self):
        """Валидация критических путей"""
        required_paths = [
            self.project_root / "airflow/dags",
            self.project_root / "src",
            self.project_root / "config"
        ]

        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path.relative_to(self.project_root)))

        if missing_paths:
            self.logger.warning(f"Отсутствуют важные директории: {missing_paths}")
            self.logger.warning(f"Некоторые функции могут работать некорректно")

    @staticmethod
    def _convert_to_wsl_path(windows_path):
        """Конвертация Windows пути в WSL путь"""
        path_str = str(windows_path)

        # Конвертация для любых дисков (C:\foo\bar -> /mnt/c/foo/bar)
        if ":" in path_str and path_str[1:3] == ":\\":
            drive_letter = path_str[0].lower()
            remaining_path = path_str[3:].replace("\\", "/")
            return f"/mnt/{drive_letter}/{remaining_path}"
        else:
            # Уже WSL путь или относительный
            return path_str.replace("\\", "/")

    @staticmethod
    def _quote_wsl_path(path: str) -> str:
        """Экранирование WSL-пути для безопастного использования в bash"""
        return shlex.quote(path)

    def _get_airflow_env(self) -> str:
        """Получение строки с переменными окружения Airflow"""
        return self.AIRFLOW_ENV_TEMPLATE.format(airflow_home=self._quote_wsl_path(self.wsl_airflow_home))

    def _build_airflow_command(self, airflow_args: str) -> str:
        """Построение полной команды для запуска Airflow в WSL"""
        return (
            f"cd {self._quote_wsl_path(self.wsl_project_root)} && "
            f"{self._get_airflow_env()} && "
            f".venv/bin/airflow {airflow_args}"
        )

    def _run_wsl_command(self, command, timeout=300, capture_output=True):
        """Выполнение WSL команд"""
        try:
            self.logger.debug(f"Выполнение WSL команды: {command[:100]}...")

            results = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=capture_output,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=timeout
            )

            if results.returncode != 0 and capture_output:
                self.logger.warning(f"Команда завершилась с кодом {results.returncode}")
                if results.stderr:
                    self.logger.error(f"Stderr: {results.stderr[:200]}")

            return results

        except subprocess.TimeoutExpired:
            self.logger.error(f"Таймаут выполнения команды: {command[:50]}...")
            return None
        except Exception as e:
            self.logger.error(f"Ошибка выполнения команды: {e}")
            return None

    def _run_wsl_command_with_retry(self, command, timeout=300, retries=3, delay=5):
        """Выполнение WSL команды с повторными попытками"""
        for attempt in range(1, retries + 1):
            result = self._run_wsl_command(command, timeout=timeout)

            if result is not None and result.returncode == 0:
               return result

            if attempt < retries:
                self.logger.warning(f"Попытка {attempt}/{retries} не удалась, повтор через {delay} сек...")

            self.logger.error(f"Команда не выполнена после {retries} попыток")
            return result

    def sync_files_to_wsl(self):
        """Синхронизация файлов из Windows в WSL"""
        self.logger.info("Синхронизация файлов с WSL")

        sync_results = {
            "successful": 0,
            "failed": 0,
            "total": 0
        }

        # Список критичных файлов для синхронизации
        critical_files = [
            {
                "name": "Основной DAG",
                "source": self.project_root / "airflow/dags/demand_forecasting_dag.py",
                "target": f"{self.wsl_airflow_home}/dags/demand_forecasting_dag.py"
            },
            {
                "name": "Мониторинг DAG",
                "source": self.project_root / "airflow/dags/monitoring/dag_monitor.py",
                "target": f"{self.wsl_airflow_home}/dags/monitoring/dag_monitor.py"
            }
        ]

        # Синхронизация критичных файлов
        for file_info in critical_files:
            source = file_info["source"]
            target = file_info["target"]
            name = file_info["name"]

            if not source.exists():
                self.logger.warning(f"Файл не найден: {source}")
                continue

            try:
                # Создаем целевую директорию
                target_dir = "/".join(target.split("/")[:-1])
                quoted_dir = self._quote_wsl_path(target_dir)
                mkdir_result = self._run_wsl_command(f"mkdir -p {quoted_dir}", timeout=30)

                if mkdir_result is None or mkdir_result.returncode != 0:
                    self.logger.error(f"Не удалось создать директорию: {target_dir}")
                    sync_results["failed"] += 1
                    continue

                # Читаем файл с обработкой кодировки
                with open(source, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Копируем файл в WSL
                quoted_target = self._quote_wsl_path(target)
                copy_command = f"cat > {quoted_target}"
                copy_proccess = subprocess.run(
                    ["wsl", "bash", "-c", copy_command],
                    input=file_content.encode("utf-8"),
                    capture_output=True,
                    timeout=30
                )

                if copy_proccess.returncode == 0:
                    self.logger.info(f"Файл синхронизирован: {name}")
                    sync_results["successful"] += 1
                else:
                    self.logger.error(f"Ошибка копирования {name} : {copy_proccess.stderr}")
                    sync_results["failed"] += 1

                sync_results["total"] += 1

            except Exception as e:
                self.logger.error(f"Ошибка синхронизации {name}: {e}")
                sync_results["failed"] += 1
                sync_results["total"] += 1

        # Рекурсивная синхронизация директорий
        directories_to_sync = [
            ("src", f"{self.wsl_project_root}/src", "**/*.py"),
            ("config", f"{self.wsl_project_root}/config", "**/*.py"),
            ("airflow/dags/monitoring", f"{self.wsl_airflow_home}/dags/monitoring", "**/*.py")
        ]

        for dir_name, wsl_target, pattern in directories_to_sync:
            source_dir = self.project_root / dir_name

            if not source_dir.exists():
                self.logger.warning(f"Директория не найдена: {source_dir}")
                continue

            try:
                # Создаем целевую директорию
                quoted_target = self._quote_wsl_path(wsl_target)
                self._run_wsl_command(f"mkdir -p {quoted_target}", timeout=30)

                # Находим все файлы
                for source_file in source_dir.rglob(pattern):
                    try:
                        rel_path = source_file.relative_to(source_dir)
                        rel_posix = str(rel_path).replace("\\", "/")
                        wsl_file_path = f"{wsl_target}/{rel_posix}"
                        wsl_file_dir = "/".join(wsl_file_path.split("/")[:-1])

                        # Создаем поддиректории
                        quoted_file_dir = self._quote_wsl_path(wsl_file_dir)
                        self._run_wsl_command(f"mkdir -p {quoted_file_dir}", timeout=10)

                        # Копируем файл
                        with open(source_file, "r", encoding="utf-8") as f:
                            content = f.read()

                        quoted_file_path = self._quote_wsl_path(wsl_file_path)
                        subprocess.run(
                            ["wsl", "bash", "-c", f"cat > {quoted_file_path}"],
                            input=content.encode("utf-8"),
                            capture_output=True,
                            timeout=30
                        )

                        sync_results["successful"] += 1
                        sync_results["total"] += 1

                    except Exception as e:
                        self.logger.warning(f"Ошибка копирования: {source_file}: {e}")
                        sync_results["failed"] += 1
                        sync_results["total"] += 1

            except Exception as e:
                self.logger.error(f"Ошибка синхронизации директории {dir_name}: {e}")

        # Итоговая статистика
        ok = sync_results["successful"]
        total = sync_results["total"]
        failed = sync_results["failed"]

        self.logger.info(f"Синхронизация завершена: {ok}/{total} файлов")

        if failed > 0:
            self.logger.warning(f"Ошибка при синхронизации: {failed}")

        return failed == 0

    def check_wsl_installation(self):
        """Проверка установки WSL"""
        self.logger.info("Проверка установки WSL")

        try:
            # Проверка наличия WSL
            version_check = subprocess.run(
                ["wsl", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if version_check.returncode == 0:
                self.logger.info(f"WSL обнаружен: {version_check.stdout.splitlines()[0]}")

                # Проверяем доступные дистрибутивы
                list_check = subprocess.run(
                    ["wsl", "-l", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if list_check.returncode == 0:
                    self.logger.debug("Доступные дистрибутивы WSL:")
                    for line in list_check.stdout.splitlines():
                        self.logger.debug(f"{line}")

                return True
            else:
                self.logger.error("WSL не установлен или не настроен")
                self.logger.error(f"Ошибка: {version_check.stderr}")
                return False

        except FileNotFoundError:
            self.logger.error("Команда wsl не найдена. Установите WSL2.")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Таймаут проверки WSL")
            return False

    def check_airflow_installation(self):
        """Проверка установки Airflow в WSL"""
        self.logger.info("Проверка установки Airflow")

        try:
            wsl_root = self._quote_wsl_path(self.wsl_project_root)
            result = self._run_wsl_command(
                f"cd {wsl_root} && "
                f"source .venv/bin/activate && "
                f"python -c \"import airflow; print('Airflow версия:', airflow.__version__)\"",
                timeout=60
            )

            if result and result.returncode == 0:
                self.logger.info(result.stdout.strip())
                return True
            else:
                self.logger.error("Airflow не установлен в WSL")
                if result and result.stderr:
                    self.logger.error(f"Ошибка: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка проверки Airflow: {e}")
            return False

    def cleanup_old_database(self):
        """Очистка старой базы данных, если существует"""
        self.logger.info("Очистка старой базы данных")

        airflow_home_q = self._quote_wsl_path(self.wsl_airflow_home)
        backup_dir = f"{self.wsl_airflow_home}/backups"
        backup_dir_q = self._quote_wsl_path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Создаем бэкап старой БД
        backup_commands = [
            f"mkdir -p {backup_dir_q}",
            f"cp {airflow_home_q}/airflow.db "
            f"{backup_dir_q}/airflow.db.backup_{timestamp} 2>/dev/null || true",
            # Оставляем только последние 5 бэкапов
            f"ls -la {backup_dir_q}/*.backup_* 2>/dev/null | "
            f"tail -n +6 | xargs rm -f 2>/dev/null || true"
        ]

        for cmd in backup_commands:
            self._run_wsl_command(cmd, timeout=30)

        # Очищаем старые файлы
        cleanup_commands = [
            f"rm -f {airflow_home_q}/airflow.db",
            f"rm -f {airflow_home_q}/airflow.cfg",
            f"rm -rf {airflow_home_q}/logs/*",
            f"rm -f /tmp/airflow_*.pid 2>/dev/null || true"
        ]

        for cmd in cleanup_commands:
            result = self._run_wsl_command(cmd, timeout=30)
            if result and result.returncode != 0:
                self.logger.warning(f"Ошибка очистки: {cmd}")

        self.logger.info("Очистка базы данных завершена")
        return True

    def _check_database_exists(self) -> bool:
        """Проверка существования файла БД Airflow"""
        airflow_home_q = self._quote_wsl_path(self.wsl_airflow_home)
        check_db = self._run_wsl_command(
            f"test -f {airflow_home_q}/airflow.db && echo 'OK'",
            timeout=30
        )
        return check_db is not None and "OK" in (check_db.stdout or "")

    def initialize_database(self):
        """Инициализация базы данных Airflow в WSL"""
        self.logger.info("Инициализация базы данных Airflow")

        try:
            # Сначала синхронизируем файлы
            if not self.sync_files_to_wsl():
                self.logger.warning("Синхронизация файлов завершилась с ошибками")

            # Очищаем старую БД, только если она существует
            if self._check_database_exists():
                self.cleanup_old_database()


            # Создаем необходимые директории
            airflow_home_q = self._quote_wsl_path(self.wsl_airflow_home)
            directories = [
                f"{airflow_home_q}/dags",
                f"{airflow_home_q}/logs",
                f"{airflow_home_q}/plugins",
                f"{airflow_home_q}/backups"
            ]

            for directory in directories:
                self._run_wsl_command(f"mkdir -p {directory}", timeout=30)

            # Инициализация БД Airflow
            result = self._run_wsl_command_with_retry(
                self._build_airflow_command("db migrate"),
                timeout=100,
                retries=2,
                delay=10
            )

            if result and result.returncode == 0:
                self.logger.info("База данных Airflow инициализирована")

                # Создание ML-пула для ограничения параллелизма
                self._run_wsl_command(
                    self._build_airflow_command("pools set ml_pool 1 'ML - 1 слот'"),
                    timeout=30
                )

                # Проверяем создание файла БД
                if self._check_database_exists():
                    self.logger.info("Файл базы данных создан")
                    return True
                else:
                    self.logger.error("Файл базы данных не создан")
                    return False
            else:
                self.logger.error("Ошибка инициализации базы данных")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
            return False

    def reserialize_dags(self):
        """Ресериализация DAG файлов"""
        self.logger.info("Ресериализация DAG файлов")

        try:
            result = self._run_wsl_command_with_retry(
                self._build_airflow_command("dags reserialize"),
                timeout=120,
                retries=2,
                delay=5
            )

            if result and result.returncode == 0:
                self.logger.info("DAG файлы успешно ресериализованы")

                # Проверяем количество DAG
                check_dags = self._run_wsl_command(
                    self._build_airflow_command("dags list 2>/dev/null | wc -l"),
                    timeout=30
                )

                if check_dags and check_dags.stdout.strip():
                    try:
                        dag_count = int(check_dags.stdout.strip())
                        self.logger.info(f"Загружено DAG: {dag_count}")
                    except ValueError:
                        self.logger.debug("Не удалось определить количество DAG")

                return True
            else:
                self.logger.error("Ошибка ресериализации DAG файлов")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка при ресериализации DAG: {e}")
            return False

    def check_dags_loaded(self):
        """Проверка загрузки DAG в WSL"""
        self.logger.info("Проверка загрузки DAG")

        try:
            result = self._run_wsl_command(
                self._build_airflow_command("dags list"),
                timeout=60
            )

            if result and result.stdout:
                dags_loaded = []
                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if line.strip() and not line.startswith("-"):
                        dags_loaded.append(stripped)

                self.logger.info(f"Загружено DAG: {len(dags_loaded)}")

                for dag in dags_loaded[:5]: # Показываем первые 5
                    self.logger.debug(f"{dag}")

                if len(dags_loaded) > 5:
                    remaining = len(dags_loaded) - 5
                    self.logger.debug(f"  ... и еще {remaining} DAG")

                # Проверяем наличие наших DAG
                target_dags = ["demand_forecasting_pipeline", "weekly_model_retraining"]
                found_dags = []

                for target_dag in target_dags:
                    if any(target_dag in dag.lower() for dag in dags_loaded):
                        found_dags.append(target_dag)

                if len(found_dags) == len(target_dags):
                    self.logger.info("Все целевые DAG загружены")
                    return True
                else:
                    missing = set(target_dags) - set(found_dags)
                    self.logger.warning(f"Отсутствуют DAG: {missing}")
                    return False
            else:
                self.logger.error("Нет вывода от команды dags list")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка проверки DAG: {e}")
            return False

    def _check_api_server_health(self) -> bool:
        """Проверка здоровья api-server"""
        try:
            response = requests.get("http://localhost:8080/api/v2/monitor/health", timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError, OSError):
            return False

    def _wait_for_api_server(self) -> bool:
        """Ожидание готовности API-сервера с retry"""
        self.logger.info("Ожидание готовности API-сервера...")
        deadline = time.time() + self.API_HEALTH_TIMEOUT

        while time.time() < deadline:
            if self._check_api_server_health():
                self.logger.info("API-сервер готов к работе")
                return True
            time.sleep(self.API_HEALTH_INTERVAL)

        self.logger.warning(f"API-сервер не ответил за {self.API_HEALTH_TIMEOUT} сек")
        return False

    def _start_service(self, service_type, command, name):
        """Запуск сервиса Airflow (api-server или scheduler)"""
        self.logger.info(f"Запуск {name}")

        try:
            process = subprocess.Popen(
                ["wsl", "bash", "-c", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Даем время на запуск
            time.sleep(5)

            if process.poll() is None:
                self.logger.info(f"{name} запущен (PID: {process.pid})")

                # Проверяем, что api-server отвечает
                if service_type == "api-server":
                    self._wait_for_api_server()

                return process
            else:
                 self.logger.error(f"{name} завершился сразу после запуска (код: {process.returncode})")
                 return None

        except Exception as e:
            self.logger.error(f"Ошибка запуска {name}: {e}")
            return None

    def start_api_server(self):
        """Запуск Airflow API Server в WSL"""
        command = self._build_airflow_command("api-server --port 8080 --host 0.0.0.0")

        return self._start_service("api-server", command, "Airflow API Server")

    def start_scheduler(self):
        """Запуск Airflow Scheduler в WSL"""
        command = self._build_airflow_command("scheduler")

        return self._start_service("scheduler", command, "Airflow Scheduler")

    @staticmethod
    def _terminate_process(process, name, timeout=15):
        """Корректное завершение процесса: SIGTERM -> ожидание -> SIGKILL"""
        logger = logging.getLogger(__name__)

        if process is None or process.poll() is not None:
            return None

        logger.info(f"Остановка {name} (PID: {process.pid})...")
        process.terminate()

        try:
            process.wait(timeout=timeout)
            logger.info(f"{name} остановлен корректно")
        except subprocess.TimeoutExpired:
            logger.warning(f"{name} не завершился за {timeout} сек, принудительная остановка (SIGKILL)")
            process.kill()
            process.wait(timeout=5)


    def run(self):
        """Основной метод запуска Airflow через WSL"""
        self.logger.info("=" * 60)
        self.logger.info("Запуск локального Airflow через WSL")
        self.logger.info("Система прогнозирования спроса - Локальный запуск Airflow через WSL")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Airflow home: {self.airflow_home}")
        self.logger.info(f"WSL project root: {self.wsl_project_root}")

        # Проверка WSL
        if not self.check_wsl_installation():
            self.logger.error("Ошибка: WSL не установлен. Установите WSL2 для продолжения")
            return False

        # Проверка Airflow
        if not self.check_airflow_installation():
            self.logger.error("Ошибка: Airflow не установлен в WSL. Запустите сначала setup_wsl.py для установки зависимостей")
            return False

        # Инициализация БД
        if not self.initialize_database():
            self.logger.error("Не удалось инициализировать базу данных")
            return False

        # Ресериализация DAG
        if not self.reserialize_dags():
            self.logger.warning("Ресериализация DAG не удалась, продолжаем...")

        # Запуск API Server и Scheduler
        api_server_process = self.start_api_server()
        if not api_server_process:
            self.logger.error("Не удалось запустить API Server")
            return False

        scheduler_process = self.start_scheduler()
        if not scheduler_process:
            self.logger.error("Не удалось запустить Scheduler")
            self._terminate_process(api_server_process, "API Server")
            return False

        # Даем время для загрузки DAG
        self.logger.info("Ожидание загрузки DAG файлов...")
        time.sleep(15)

        # Проверяем загрузку DAG
        if not self.check_dags_loaded():
            self.logger.warning("DAG не загружен, попробуем ресериализовать еще раз...")
            time.sleep(5)
            self.reserialize_dags()
            time.sleep(10)
            self.check_dags_loaded()

        self.logger.info("AIRFLOW УСПЕШНО ЗАПУЩЕН")
        self.logger.info("API Server: http://localhost:8080")
        self.logger.info("Для остановки нажмите Ctrl+C")


        try:
            while True:
                # Проверяем оба процесса каждые 5 секунд
                api_alive = api_server_process.poll() is None
                sched_alive = scheduler_process.poll() is None

                if not api_alive:
                    self.logger.error(f"API Server неожиданно завершился (код: {api_server_process.returncode})")
                    self._terminate_process(scheduler_process, "Scheduler")
                    return False

                if not sched_alive:
                    self.logger.error(f"Scheduler неожиданно завершился (код: {scheduler_process.returncode})")
                    self._terminate_process(api_server_process, "API Server")
                    return False

                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("Остановка Airflow...")
            self._terminate_process(scheduler_process, "Scheduler")
            self._terminate_process(api_server_process, "API Server")
            self.logger.info("Airflow остановлен")

        return True


def main():
    """Точка входа скрипта"""
    try:
        runner = AirflowRunner()
        success = runner.run()

        if success:
            runner.logger.info("Airflow завершил работу")
        else:
            runner.logger.error("Ошибка запуска Airflow")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.getLogger("airflow_runner").info("Запуск прерван пользователем")
        sys.exit(0)
    except Exception as e:
        logging.getLogger("airflow_runner").error(f"Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()
