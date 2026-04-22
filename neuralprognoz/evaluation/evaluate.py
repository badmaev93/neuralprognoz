from pathlib import Path

from neuralprognoz.evaluation import get_tester
from neuralprognoz.utils.config import Config


def start_evaluation(cfg: Config, run_dir: Path, epoch: int = None, period: str = "test"):
    """Запуск оценки обученной сети

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска, считанная из директории запуска.
    run_dir : Path
        Путь к директории запуска.
    epoch : int, optional
        Определить конкретную эпоху для оценки. По умолчанию используются веса последней эпохи.
    period : {'train', 'validation', 'test'}, optional
        Период для оценки, по умолчанию 'test'.

    """
    tester = get_tester(cfg=cfg, run_dir=run_dir, period=period, init_model=True)
    tester.evaluate(epoch=epoch, save_results=True, save_all_output=cfg.save_all_output, metrics=cfg.metrics)
