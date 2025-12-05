import subprocess
import sys
from pathlib import Path


class WSLSetup:
    """Класс для настройки WSL окружения"""
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.wsl_path = self._convert_to_wsl_path(self.project_root)

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

    def _run_wsl_command(self, command, timeout=900):
        """Выполнение WSL команд"""
        try:
            subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка выполнения команды: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("Таймаут выполнения команды")
            return False

    def check_wsl_status(self):
        """Проверка статуса WSL"""
        print("Проверка установки WSL...")

        try:
            subprocess.run(["wsl", "--status"], capture_output=True, check=True, timeout=30)
            print("WSL статус проверен")
            return True
        except subprocess.CalledProcessError:
            print("Ошибка проверки WSL")
            return False
        except subprocess.TimeoutExpired:
            print("Таймаут проверки WSL")
            return False

    def setup_python_environment(self):
        """Настройка Python окружения"""
        print("Настройка Python окружения...")

        # Создаем виртуальное окружение
        if not self._run_wsl_command(f"cd {self.wsl_path} && python3 -m venv .venv", timeout=300):
            return False

        print("Python окружение настроено")
        return True

    def install_python_dependencies(self):
        """Установка Python зависимостей"""
        print("Установка Python зависимостей...")

        # Устанавливаем зависимости через pip из venv
        command = f"cd {self.wsl_path} && .venv/bin/pip install -r requirements.txt"
        if not self._run_wsl_command(command, timeout=900):
            return False

        print("Python зависимости установлены")
        return True

    def run_setup(self):
        """Основной метод настройки"""
        print("Настройка WSL окружения для системы прогнозирования спроса")

        if not self.check_wsl_status():
            print("WSL не установлен. Установите WSL2 перед продолжением.")
            return False

        if not self.setup_python_environment():
            print("Ошибка настройки Python окружения")
            return False

        if not self.install_python_dependencies():
            print("Ошибка установки Python зависимостей")
            return False

        print("Настройка WSL окружения завершена успешно")
        return True


def main():
    """Точка входа скрипта настройки"""
    setup = WSLSetup()

    success = setup.run_setup()

    if success:
        print("WSL окружение готово к использованию")
    else:
        print("Ошибка настройки WSL окружения")
        sys.exit(1)


if __name__ == "__main__":
    main()