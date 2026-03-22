from neuralprognoz.training.basetrainer import BaseTrainer
from neuralprognoz.utils.config import Config


def start_training(cfg: Config):
    """Запуск обучения модели.
    
    Параметры.
    ----------
    cfg : Config
        Конфигурация запуска.

    """
    # Отдельный случай - MC-LSTM: головной модуль возвращает пустую строку,
    # однако сама модель обучается как регрессионная модель.
    if cfg.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Не распознана регрессионная голова {cfg.head}.")
    trainer.initialize_training()
    trainer.train_and_validate()
