import subprocess
import sys
import time
from pathlib import Path


class AirflowRunner:
    """Класс для управления локальным запуском Airflow через WSL"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.airflow_home = self.project_root / "airflow"
        self.wsl_project_root = self._convert_to_wsl_path(self.project_root)
        self.wsl_airflow_home = self.wsl_project_root + "/airflow"

    def _convert_to_wsl_path(self, windows_path):
        """Конвертация Windows пути в WSL путь"""
        path_str = str(windows_path)

        # Убираем префикс пути Windows
        if path_str.startswith("C:\\"):
            return "/mnt/c/" + path_str[3:].replace("\\", "/")
        elif path_str.startswith("D:\\"):
            return "/mnt/d/" + path_str[3:].replace("\\", "/")
        elif path_str.startswith("E:\\"):
            return "/mnt/e/" + path_str[3:].replace("\\", "/")
        elif ":" in path_str:
            drive_letter = path_str[0].lower()
            return f"/mnt/{drive_letter}/" + path_str[3:].replace("\\", "/")
        else:
            return path_str.replace("\\", "/")

    def _run_wsl_command(self, command, timeout=300):
        """Выполнение WSL команд"""
        try:
            results = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
            return results
        except subprocess.CalledProcessError as e:
            print(f"Ошибка выполнения команды: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("Таймаут выполнения команды")
            return None

    def sync_files_to_wsl(self):
        """Сохранение файлов из Windows в WSL"""
        print("Сохранение файлов в WSL")

        # Главный DAG файл
        dag_file = self.project_root / "airflow/dags/demand_forecasting_dag.py"
        wsl_dag_path = f"{self.wsl_airflow_home}/dags/demand_forecasting_dag.py"

        if dag_file.exists():
            print(f"Копируем DAG файл: {dag_file.name}")
            try:
                # Читаем файл с UTF-8
                with open(dag_file, "r", encoding="utf-8") as f:
                    dag_content = f.read()

                # Создаем директорию в WSL
                self._run_wsl_command(f"mkdir -p {self.wsl_airflow_home}/dags")

                # Копируем файл в WSL с UTF-8
                subprocess.run([
                    "wsl", "bash", "-c",
                    f"cat > {wsl_dag_path}"
                ], input=dag_content.encode("utf-8"), check=True, timeout=30)

                print(f"DAG файл сохранен в WSL: {wsl_dag_path}")
            except Exception as e:
                print(f"Ошибка сохранения DAG файла: {e}")

        # Папка src
        src_dir = self.project_root / "src"
        wsl_src_dir = f"{self.wsl_project_root}/src"

        if src_dir.exists():
            print(f"Копируем папку src...")
            try:
                # Копируем только .py файлы
                for file_path in src_dir.rglob("*.py"):
                    rel_path = file_path.relative_to(src_dir)
                    # Исправляем путь: заменяем все обратные слеши
                    rel_path_str = str(rel_path).replace("\\", "/").replace("\\\\", "/")
                    wsl_file_path = f"{wsl_src_dir}/{rel_path_str}"
                    wsl_file_dir = "/".join(wsl_file_path.split("/")[:-1])

                    # Создаем директорию
                    self._run_wsl_command(f"mkdir -p {wsl_file_dir}")

                    # Читаем и копируем файл
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                    subprocess.run([
                        "wsl", "bash", "-c",
                        f"cat > {wsl_file_path}"
                    ], input=file_content.encode("utf-8"), check=True, timeout=30)

                print(f"Папка src сохранена в WSL: {wsl_src_dir}")
            except Exception as e:
                print(f"Ошибка сохранения папки src: {e}")

        # Папка monitoring
        monitoring_dir = self.project_root / "monitoring"
        wsl_monitoring_dir = f"{self.wsl_project_root}/monitoring"

        if monitoring_dir.exists():
            print(f"Копируем папку monitoring...")
            try:
                self._run_wsl_command(f"rm -rf {wsl_monitoring_dir}")
                self._run_wsl_command(f"mkdir -p {wsl_monitoring_dir}")

                # Копируем .py файлы из monitoring
                for file_path in monitoring_dir.rglob("*.py"):
                    rel_path = file_path.relative_to(monitoring_dir)
                    # Исправляем путь: заменяем все обратные слеши
                    rel_path_str = str(rel_path).replace("\\", "/")
                    wsl_file_path = f"{wsl_monitoring_dir}/{rel_path_str}"
                    wsl_file_dir = "/".join(wsl_file_path.split("/")[:-1])

                    self._run_wsl_command(f"mkdir -p {wsl_file_dir}")

                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                    subprocess.run([
                        "wsl", "bash", "-c",
                        f"cat > {wsl_file_path}"
                    ], input=file_content.encode("utf-8"), check=True, timeout=30)

                print(f"Папка monitoring сохранена в WSL: {wsl_monitoring_dir}")
            except Exception as e:
                print(f"Ошибка сохранения папки monitoring: {e}")

        print("Все файлы сохранены в WSL")
        return True

    def check_wsl_installation(self):
        """Проверка установки WSL"""
        try:
            subprocess.run(["wsl", "--status"], capture_output=True, check=True, timeout=30)
            print("WSL установлен и работает")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"Ошибка проверки WSL: {e}")
            return False

    def check_airflow_installation(self):
        """Проверка установки Airflow в WSL"""
        try:
            result = subprocess.run([
                "wsl", "bash", "-c",
                f"cd {self.wsl_project_root} && .venv/bin/airflow version"
            ], capture_output=True, text=True, check=True, timeout=30)

            print(f"Airflow версия: {result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"Ошибка проверки Airflow: {e}")
            return False

    def cleanup_old_database(self):
        """Очистка старой базы данных, если существует"""
        print("Очистка старой базы данных...")

        # Удаляем старые файлы БД
        cleanup_commands = [
            f"rm -f {self.wsl_airflow_home}/airflow.db",
            f"rm -f {self.wsl_airflow_home}/webserver_config.py",
            f"rm -f {self.wsl_airflow_home}/airflow.cfg",
            "rm -f /home/oldoin/airflow/airflow.db"  # Очищаем и дефолтный путь
        ]

        for cmd in cleanup_commands:
            self._run_wsl_command(cmd, timeout=30)

    def initialize_database(self):
        """Инициализация базы данных Airflow в WSL"""
        print("Инициализация базы данных Airflow")

        try:
            # Сначала сохраняем файлы в WSL
            self.sync_files_to_wsl()

            # Очистка старой БД
            self.cleanup_old_database()

            # Создание необходимых директорий в WSL
            self._run_wsl_command(f"mkdir -p {self.wsl_airflow_home}/dags", timeout=30)
            self._run_wsl_command(f"mkdir -p {self.wsl_airflow_home}/logs", timeout=30)
            self._run_wsl_command(f"mkdir -p {self.wsl_airflow_home}/plugins", timeout=30)

            # Инициализация БД Airflow
            result = self._run_wsl_command(
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f"export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
                f"export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='sqlite:///{self.wsl_airflow_home}/airflow.db?journal_mode=DELETE&locking_mode=EXCLUSIVE' && "
                f".venv/bin/airflow db migrate",
                timeout=120)

            if result and result.returncode == 0:
                print("База данных Airflow инициализирована")
                return True
            else:
                print(f"Ошибка инициализации базы данных")
                if result:
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"Ошибка инициализации базы данных: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("Таймаут инициализации базы данных")
            return False

    def reserialize_dags(self):
        """Ресериализация DAG файлов"""
        print("Ресериализация DAG файлов...")

        try:
            result = self._run_wsl_command(
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f"export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
                f".venv/bin/airflow dags reserialize",
                timeout=120
            )

            if result and result.returncode == 0:
                print("DAG файлы успешно ресериализованы")
                print(f"Вывод: {result.stdout[:200]}...")  # Показываем первые 200 символов
                return True
            else:
                print("Ошибка ресериализации DAG файлов")
                if result:
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                return False

        except Exception as e:
            print(f"Ошибка при ресериализации DAG: {e}")
            return False

    def start_api_server(self):
        """Запуск Airflow API Server в WSL"""
        print("Запуск Airflow API Server")

        try:
            command = (
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f"export AIRFLOW__CORE__DAGS_FOLDER={self.wsl_airflow_home}/dags && "
                f"export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
                f"export AIRFLOW__CORE__EXECUTOR=LocalExecutor && "
                f"export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='sqlite:///{self.wsl_airflow_home}/airflow.db?journal_mode=DELETE&locking_mode=EXCLUSIVE' && "
                f"export AIRFLOW__LOGGING__BASE_LOG_FOLDER={self.wsl_airflow_home}/logs && "
                f".venv/bin/airflow api-server --port 8080 --host 0.0.0.0")

            process = subprocess.Popen(
                ["wsl", "bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(15)

            if process.poll() is None:
                print("Airflow API Server запущен: http://localhost:8080")
                return process
            else:
                print("API Server завершился сразу после запуска")
                return None

        except Exception as e:
            print(f"Ошибка запуска API Server: {e}")
            return None

    def start_scheduler(self):
        """Запуск Airflow Scheduler в WSL"""
        print("Запуск Airflow Scheduler")

        try:
            command = (
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f"export AIRFLOW__CORE__DAGS_FOLDER={self.wsl_airflow_home}/dags && "
                f"export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
                f"export AIRFLOW__CORE__EXECUTOR=LocalExecutor && "
                f"export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='sqlite:///{self.wsl_airflow_home}/airflow.db?journal_mode=DELETE&locking_mode=EXCLUSIVE' && "
                f"export AIRFLOW__LOGGING__BASE_LOG_FOLDER={self.wsl_airflow_home}/logs && "
                f".venv/bin/airflow scheduler")

            process = subprocess.Popen(
                ["wsl", "bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(10)

            if process.poll() is None:
                print("Airflow Scheduler запущен")
                return process
            else:
                print("Scheduler завершился сразу после запуска")
                return None

        except Exception as e:
            print(f"Ошибка запуска Scheduler: {e}")
            return None

    def check_dags_loaded(self):
        """Проверка загрузки DAG в WSL"""
        print("Проверка загрузки DAG")

        try:
            result = self._run_wsl_command(
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f"export AIRFLOW__CORE__LOAD_EXAMPLES=false && "
                f".venv/bin/airflow dags list",
                timeout=60
            )

            if result and result.stdout:
                print("Список доступных DAG:")
                print(result.stdout)

                # Проверяем наличие конкретных DAG
                if "demand_forecasting" in result.stdout.lower():
                    print("DAG demand_forecasting загружен")
                    return True
                elif "demand_forecasting_pipeline" in result.stdout.lower():
                    print("DAG demand_forecasting_pipeline загружен")
                    return True
                else:
                    print("DAG не найден в списке")
                    return False
            else:
                print("Нет вывода от команды dags list")
                return False

        except Exception as e:
            print(f"Ошибка проверки DAG: {e}")
            return False

    def trigger_dag(self, dag_id="demand_forecasting"):
        """Запуск DAG"""
        print(f"Запуск DAG {dag_id}")

        try:
            result = subprocess.run([
                "wsl", "bash", "-c",
                f"cd {self.wsl_project_root} && "
                f"export AIRFLOW_HOME={self.wsl_airflow_home} && "
                f".venv/bin/airflow dags trigger {dag_id}"
            ], capture_output=True, text=True, check=True, timeout=60)

            print(f"DAG {dag_id} успешно запущен")
            print(f"Результат: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Ошибка запуска DAG: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False

    def run(self):
        """Основной метод запуска Airflow через WSL"""
        print("Запуск локального Airflow через WSL")

        if not self.check_wsl_installation():
            print("Ошибка: WSL не установлен")
            print("Установите WSL2 для продолжения")
            return False

        if not self.check_airflow_installation():
            print("Ошибка: Airflow не установлен в WSL")
            print("Запустите сначала setup_wsl.py для установки зависимостей")
            return False

        if not self.initialize_database():
            return False

        if not self.reserialize_dags():
            print("Предупреждение: ресериализация DAG не удалась, продолжаем...")

        # Запуск API Server и Scheduler
        api_server_process = self.start_api_server()
        if not api_server_process:
            return False

        scheduler_process = self.start_scheduler()
        if not scheduler_process:
            api_server_process.terminate()
            return False

        # Даем время для загрузки DAG
        print("\nОжидание загрузки DAG файлов...")
        time.sleep(15)

        # Проверяем загрузку DAG
        if not self.check_dags_loaded():
            print("Предупреждение: DAG не загружен, попробуем ресериализовать еще раз...")
            time.sleep(5)
            self.reserialize_dags()
            time.sleep(10)
            self.check_dags_loaded()

        print("AIRFLOW УСПЕШНО ЗАПУЩЕН:")
        print("API Server: http://localhost:8080")
        print("Для остановки нажмите Ctrl+C")

        try:
            api_server_process.wait()
            scheduler_process.wait()
        except KeyboardInterrupt:
            print("\nОстановка Airflow")
            api_server_process.terminate()
            scheduler_process.terminate()
            print("Airflow остановлен")

        return True


def main():
    """Точка входа скрипта"""
    runner = AirflowRunner()

    print("Система прогнозирования спроса - Локальный запуск Airflow через WSL")
    print("Конфигурация:")
    print(f"Project root: {runner.project_root}")
    print(f"Airflow home: {runner.airflow_home}")
    print(f"DAGs folder: {runner.airflow_home / "dags"}")
    print(f"WSL project root: {runner.wsl_project_root}")

    success = runner.run()

    if success:
        print("Airflow завершил работу")
    else:
        print("Ошибка запуска Airflow")
        sys.exit(1)


if __name__ == "__main__":
    main()