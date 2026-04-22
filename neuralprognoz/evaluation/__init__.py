from pathlib import Path

from neuralprognoz.evaluation.tester import BaseTester, RegressionTester, UncertaintyTester
from neuralprognoz.utils.config import Config


def get_tester(cfg: Config, run_dir: Path, period: str, init_model: bool) -> BaseTester:
    """Получить конкретные объекты класса тестера в зависимости от типа модели (головы).
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    run_dir : Path
        Путь к директории запуска.
    period : {'train', 'validation', 'test'}
        Период для оценки.
    init_model : bool
        Если True, веса модели будут инициализированы контрольной точкой из последней доступной эпохи в `run_dir`.

    Возвращает
    -------
    BaseTester
        `RegressionTester` если голова модели 'regression'. `UncertaintyTester` если голова модели одна из 
        {'gmm', 'cmal', 'umal'} или если оценка выполняется в режиме MC-Dropout.
    """
    if cfg.mc_dropout or cfg.head.lower() in ["gmm", "cmal", "umal"]:
        Tester = UncertaintyTester
    # MC-LSTM — это особый случай, когда голова возвращает пустую строку, но модель обучается как регрессионная модель.
    elif cfg.head.lower() in ["regression", ""]:
        Tester = RegressionTester
    else:
        NotImplementedError(f"Метод оценки для {cfg.head} не реализован")

    return Tester(cfg=cfg, run_dir=run_dir, period=period, init_model=init_model)
