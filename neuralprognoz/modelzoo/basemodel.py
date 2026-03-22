from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.datautils.utils import load_scaler
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.samplingutils import sample_pointpredictions, umal_extend_batch


class BaseModel(nn.Module):
    """Абстрактный класс базовой модели, для тренировки этот класс не использовать.

    Использовать субклассы класса для тренировки/оценки разных моделей, например `CudaLSTM` для тренировки стандартной
    LSTM модели или `EA-LSTM` для тренировки LSTM, учитывающей сущности. Обратиться к :doc:`Documentation/Modelzoo </usage/models>`
    за полным списком доступных моделей и за тем, как интегрировать новую модель.   


    Параметры
    ----------
    cfg : Config
        Конфигурация запуска
    """
    # Спецификация субмоделей модели которая позже может быть использована для файнтюнинга. Названия должны соответствовать атрибутам класса.
    module_parts = []

    def __init__(self, cfg: Config):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.output_size = len(cfg.target_variables)
        if cfg.head.lower() == 'gmm':
            self.output_size *= 3 * cfg.n_distributions
        elif cfg.head.lower() == 'cmal':
            self.output_size *= 4 * cfg.n_distributions
        elif cfg.head.lower() == 'umal':
            self.output_size *= 2

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Предоставление выборки точечных прогнозов из вероятностной модели.
        
        Обертка функции `sample_pointpredictions`, которая предоставляет разные точки сэмлинга функций для 
        разных подходов к оценке неопределенности. Также - опция для обработки выборок 
        с отрицательными точками прогнозирования, которые возникают при выборке из оценок неопределенности. 
        Управление с помощью конфигурации.
             
        Параметры
        ----------
        data : Dict[str, torch.Tensor]
            Словарь, содержит входные признаки как пары ключ-значение.
        n_samples : int
            Количество точек прогноза, которые не выбираются из модели. 

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Выборочные точечные прогнозы 
        """
        scaler = load_scaler(self.cfg.run_dir)
        return sample_pointpredictions(self, data, n_samples, scaler)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Проход вперед.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержит входные признаки как пары ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Модельный вывод и, возможно, любые промежуточные состояния и активации в виде словаря.
        """
        raise NotImplementedError

    def pre_model_hook(self, data: Dict[str, torch.Tensor], is_train: bool) -> Dict[str, torch.Tensor]:
        """Функция, выполняемая перед моделью при обучении, валидации и тестировании. 
        Может быть адаптировано в зависимости от конфигурации запуска и предоставленных аргументов.

        Параметры
        ----------
        data : Dict[str, torch.Tensor]
            Словарь, содержащий входные данные в виде пар ключ-значение и меток y.
        is_train : bool
            Определяет, выполняется ли перехват в режиме обучения или в режиме проверки/тестирования.

        Возвращает
        -------
        data : Dict[str, torch.Tensor]
            Измененные (или неизмененные) данные, которые используются для обучения или оценки.
        """
        if self.cfg.head.lower() == "umal":
            data = umal_extend_batch(data, self.cfg, n_taus=self.cfg.n_taus, extend_y=True)
        else:
            # тут - можно реализовать дополнительные предварительные model hooks
            pass

        return data