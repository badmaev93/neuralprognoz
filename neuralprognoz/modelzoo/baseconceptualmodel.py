from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
from neuralprognoz.utils.config import Config


class BaseConceptualModel(nn.Module):
    """Абстрактный базовый класс модели. Не используйте этот класс для обучения модели.

    Цель — обеспечить наличие некоторых общих операций, необходимых всем концептуальным моделям. Использовать подклассы этого класса
    для обучения/оценки различных концептуальных моделей, например, `SHM`.
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    """

    def __init__(self, cfg: Config):
        super(BaseConceptualModel, self).__init__()
        self.cfg = cfg
        # Проверить, находятся ли dynamic_conceptual_inputs и target_variables в пользовательской нормализации.
        # Это необходимо, поскольку концептуальные модели консервативны по массе.
        if any(item not in cfg.custom_normalization for item in cfg.dynamic_conceptual_inputs + cfg.target_variables):
            raise RuntimeError("dynamic_conceptual_inputs и target_variables требуются в custom_normalization")

    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    def _get_dynamic_parameters_conceptual(self, lstm_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Сопоставить выходные данные управляемой данными части с предопределенными диапазонами используемой концептуальной модели.

        Параметры
        ----------
        lstm_out: torch.Tensor
            Тензор размером [batch_size, time_steps, n_param], который будет сопоставлен с предопределенными диапазонами
            параметров концептуальной модели, чтобы действовать как динамическая параметризация.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Динамическая параметризация концептуальной модели.
        """
        dynamic_parameters = {}
        for index, (parameter_name, parameter_range) in enumerate(self.parameter_ranges.items()):
            range_t = torch.tensor(parameter_range, dtype=torch.float32, device=lstm_out.device)
            range_t = range_t.repeat(lstm_out.shape[0], 1)  # Для параллельного запуска всех элементов партии
            dynamic_parameters[parameter_name] = range_t[:, :1] + torch.sigmoid(lstm_out[:, :, index]) * (range_t[:, 1:] - range_t[:, :1])

        return dynamic_parameters

    def _initialize_information(self, conceptual_inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Инициализируем структуры для хранения временной эволюции внутренних состояний и выходных потоков концептуальной
        модели

        Параметры
        ----------
        conceptual_inputs: torch.Tensor
            Входные данные концептуальной модели (динамические форсинги)

        Возвращает
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            - states: Dict[str, torch.Tensor]
                Словарь для хранения временной эволюции внутренних состояний (корзин) концептуальной модели
            - q_out: torch.Tensor
                Тензор для хранения результатов концептуальной модели
        """

        states = {}
        # инициализируем словарь для хранения эволюции состояний
        for name, value in self.initial_states.items():
            states[name] = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32,
                                       device=conceptual_inputs.device)

        # инициализируйте векторы для хранения эволюции выходных данных
        out = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1], len(self.cfg.target_variables)),
                          dtype=torch.float32, device=conceptual_inputs.device)

        return states, out


    @property
    def initial_states(self):
        raise NotImplementedError

    @property
    def parameter_ranges(self):
        raise NotImplementedError

