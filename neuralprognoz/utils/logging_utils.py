import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: str):
    """Инициализирует ведение журнала в файл `log_file` и стандартный вывод.

    Параметры
    ----------
    log_file : str
        Имя файла, в который будет вестись журнал.
    """
    # Отключение colorama для Windows
    if sys.platform == 'win32':
        os.environ['NO_COLOR'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        sys.modules['colorama'] = None
        sys.modules['colorama.ansitowin32'] = None
        sys.modules['colorama.win32'] = None
        sys.modules['colorama.initialise'] = None
        
        try:
            import colorama
            if hasattr(colorama, 'deinit'):
                colorama.deinit()
        except (ImportError, AttributeError):
            pass
    
    # Очистка существующих обработчиков журналов, чтобы избежать конфликтов.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(filename=log_file, encoding='utf-8')
    
    # Создать безопасный обработчик вывода в стандартный поток (stdout).
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Для Windows,обеспечить чистый вывод  ASCII
                if sys.platform == 'win32':
                    msg = msg.encode('ascii', 'replace').decode('ascii')
                self.stream.write(msg + '\n')
                self.flush()
            except Exception:
                # Если что-то пойдет не так, просто пропустить вывод в консоль
                pass
    
    stdout_handler = SafeStreamHandler(sys.stdout)
    
    # Простой формат для Windows
    format_str = "%(asctime)s - %(message)s"
    
    # Настройка логгирования
    logging.basicConfig(
        handlers=[file_handler, stdout_handler], 
        level=logging.INFO, 
        format=format_str,
        force=True  # принудительная реконфигарация
    )

    def exception_logging(type, value, tb):
        LOGGER.exception(f"Uncaught exception", exc_info=(type, value, tb))

    sys.excepthook = exception_logging

    LOGGER.info(f"Логгирование в {log_file} инициализировано.")


def get_git_hash() -> Optional[str]:
    """Получение хеша коммита проекта, если это репозиторий Git.

    Возвращает
    -------
    Optional[str]
        Хеш коммита Git, если проект находится в репозитории Git, иначе None.
    """
    # получение хеша коммита, если репозиторий git
    current_dir = str(Path(__file__).absolute().parent)
    try:
        if subprocess.call(["git", "-C", current_dir, "branch"], stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0:
            return subprocess.check_output(["git", "-C", current_dir, "describe", "--always"]).strip().decode('ascii')
    except OSError:
        return None  # вероятно, Git не установлен.

def save_git_diff(run_dir: Path):
    """Попытка сохранить изменения Git в файл.

    Параметры
    ----------
    run_dir : Path
        Директория текущего запуска
    """
    base_dir = str(Path(__file__).absolute().parent)
    try:
        # изменения включают как подготовленные, так и неподготовленные, поэтому используется "HEAD".
        out = subprocess.check_output(['git', '-C', base_dir, 'diff', 'HEAD'], stderr=subprocess.DEVNULL)
    except OSError:
        LOGGER.warning('Не удалось сохранить изменения Git, вероятно потому, что Git не установлен. '
                       'или, потому что установленная версия Git слишком старая (< 1.8.5)')
        return

    new_diff = out.strip().decode('utf-8')

    if new_diff:
        existing_diffs = list(run_dir.glob('neuralprognoz*.diff'))
        if len(existing_diffs) > 0:
            last_diff_path = run_dir / f'neuralprognoz-{len(existing_diffs) - 1}.diff'
            with last_diff_path.open('r') as last_diff_file:
                last_diff = last_diff_file.read()
            if last_diff == new_diff:
                LOGGER.info(f'Репозиторий Git содержит незакоммитированные изменения, которые хранятся в... {last_diff_path}.')
                return

        file_path = run_dir / f'neuralprognoz-{len(existing_diffs)}.diff'
        LOGGER.warning(f'Репозиторий Git содержит незакоммиченные изменения. Записываю дифф в... {file_path}.')
        with file_path.open('w') as diff_file:
            diff_file.write(new_diff)
