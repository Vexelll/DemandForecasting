import logging
import subprocess
import sys
from pathlib import Path


class WSLSetup:
    """Класс для настройки WSL окружения"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.wsl_path = self._convert_to_wsl_path(self.project_root)
        self.logger = self._setup_logger()

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
            log_file = self.project_root / "logs/wsl_setup.log"
            log_file.parent.mkdir(exist_ok=True, parents=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _convert_to_wsl_path(self, windows_path):
        """Конвертация Windows пути в WSL путь"""
        path_str = str(windows_path)

        # Конвертация для любых дисков
        if ":" in path_str and path_str[1:3] == ":\\":
            drive_letter = path_str[0].lower()
            remaining_path = path_str[3:].replace("\\", "/")
            return f"/mnt/{drive_letter}/{remaining_path}"
        else:
            # Уже WSL путь или относительный
            return path_str.replace("\\", "/")

    def _run_wsl_command(self, command, timeout=900):
        """Выполнение WSL команд"""
        try:
            self.logger.debug(f"Выполнение WSL команды: {command[:100]}...")

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
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка выполнения команды: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            return False

    def check_wsl_status(self):
        """Проверка статуса WSL"""
        self.logger.info("Проверка установки WSL...")

        try:
            subprocess.run(["wsl", "--status"], capture_output=True, check=True, timeout=30)
            self.logger.info("WSL статус проверен")
            return True

        except subprocess.CalledProcessError:
            self.logger.error("Ошибка проверки WSL")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Таймаут проверки WSL")
            return False

    def setup_python_environment(self):
        """Настройка Python окружения"""
        self.logger.info("Настройка Python окружения...")

        # Создаем виртуальное окружение
        if not self._run_wsl_command(f"cd {self.wsl_path} && python3 -m venv .venv", timeout=300):
            return False

        self.logger.info("Python окружение настроено")
        return True

    def install_python_dependencies(self):
        """Установка Python зависимостей"""
        self.logger.info("Установка Python зависимостей...")

        # Устанавливаем зависимости через pip из venv
        command = f"cd {self.wsl_path} && .venv/bin/pip install -r requirements.txt"
        if not self._run_wsl_command(command, timeout=1000):
            return False

        self.logger.info("Python зависимости установлены")
        return True

    def run_setup(self):
        """Основной метод настройки"""
        self.logger.info("Настройка WSL окружения для системы прогнозирования спроса")

        if not self.check_wsl_status():
            self.logger.error("WSL не установлен. Установите WSL2 перед продолжением")
            return False

        if not self.setup_python_environment():
            self.logger.error("Ошибка настройки Python окружения")
            return False

        if not self.install_python_dependencies():
            self.logger.error("Ошибка установки Python зависимостей")
            return False

        self.logger.info("Настройка WSL окружения завершена успешно")
        return True


def main():
    """Точка входа скрипта настройки"""
    try:
        setup = WSLSetup()
        success = setup.run_setup()

        if success:
            print("WSL окружение готово к использованию")
        else:
            print("Ошибка настройки WSL окружения")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.getLogger("wsl_setup").info("Настройка прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.getLogger("wsl_setup").error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
