import logging
import sys
import subprocess
from pathlib import Path


class WSLSetup:
    """Настройка WSL окружения: venv + зависимости + Airflow init + структура директорий"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.wsl_path = self._convert_to_wsl_path(self.project_root)

    @staticmethod
    def _convert_to_wsl_path(windows_path: Path) -> str:
        """C\\foo\\bar -> /mnt/с/foo/bar"""
        path_str = str(windows_path)

        if ":" in path_str and path_str[1:3] == ":\\":
            drive_letter = path_str[0].lower()
            remaining_path = path_str[3:].replace("\\", "/")
            return f"/mnt/{drive_letter}/{remaining_path}"

        # Уже WSL путь или относительный
        return path_str.replace("\\", "/")

    def _run_wsl_command(self, command: str, timeout: int = 900) -> bool:
        """Выполнение WSL команды, True = успех"""
        try:
            self.logger.debug(f"WSL: {command[:100]}...")

            result = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )

            if result.returncode != 0:
                self.logger.error(f"Ошибка выполнения команды: {result.stderr[:200]}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"Таймаут выполнения команды")
            return False
        except FileNotFoundError:
            self.logger.error(f"wsl.exe не найден - WSL не установлен")
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            return False

    def check_wsl_status(self) -> bool:
        """Проверка установки WSL"""
        self.logger.info("Проверка установки WSL...")

        try:
            result = subprocess.run(
                ["wsl", "--status"],
                capture_output=True,
                check=True,
                timeout=30
            )

            if result.returncode != 0:
                self.logger.error("WSL не готов к работе")
                return False

            self.logger.info("WSL статус проверен")
            return True

        except FileNotFoundError:
            self.logger.error("wsl.exe не найден - WSL не установлен")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Таймаут проверки WSL")
            return False

    def setup_directory_structure(self) -> bool:
        """Создание структуры директорий в WSL: airflow/dags/monitoring, src, config, logs"""
        self.logger.info("Создание структуры директорий...")

        dirs = [
            f"{self.wsl_path}/airflow/dags/monitoring",
            f"{self.wsl_path}/airflow/logs",
            f"{self.wsl_path}/data/raw",
            f"{self.wsl_path}/data/processed",
            f"{self.wsl_path}/data/outputs",
            f"{self.wsl_path}/models",
            f"{self.wsl_path}/reports",
            f"{self.wsl_path}/logs"
        ]

        mkdir_cmd = "mkdir -p " + " ".join(dirs)
        if not self._run_wsl_command(mkdir_cmd, timeout=30):
            return False

        self.logger.info("Структура директорий создана")
        return True

    def setup_python_environment(self) -> bool:
        """Настройка виртуального окружения"""
        self.logger.info("Настройка Python окружения...")

        if not self._run_wsl_command(f"cd {self.wsl_path} && python3 -m venv .venv", timeout=300):
            return False

        self.logger.info("Python окружение настроено")
        return True

    def install_python_dependencies(self) -> bool:
        """pip install -r requirements.txt в venv"""
        self.logger.info("Установка Python зависимостей...")

        command = f"cd {self.wsl_path} && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt"
        if not self._run_wsl_command(command, timeout=1000):
            return False

        self.logger.info("Python зависимости установлены")
        return True

    def run_setup(self) -> bool:
        """Полная настройка: WSL -> директории -> venv -> pip"""
        self.logger.info("Настройка WSL окружения для системы прогнозирования спроса")

        steps = [
            ("Проверка WSL", self.check_wsl_status),
            ("Структура директорий", self.setup_directory_structure),
            ("Python окружение", self.setup_python_environment),
            ("Python зависимости", self.install_python_dependencies)
        ]

        for step_name, step_fn in steps:
            if not step_fn():
                self.logger.error(f"ошибка на шаге: {step_name}")
                return False
            self.logger.info(f"Шаг завершен: {step_name}")

        self.logger.info("Настройка WSL окружения завершена")
        return True

def main():
    """Точка входа скрипта настройки"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        setup = WSLSetup()
        success = setup.run_setup()

        if success:
            print("WSL окружение готово к использованию")
        else:
            print("Ошибка настройки WSL окружения")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Настройка прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
