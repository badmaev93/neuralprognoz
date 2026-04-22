import warnings
from builtins import FutureWarning
from typing import Dict

import torch

from neuralprognoz.modelzoo.cudalstm import CudaLSTM
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config


class EmbCudaLSTM(BaseModel):
    """EmbCudaLSTM модель, которая добавляет embedding сети для статических входов к стандартному LSTM.

    .. deprecated:: 0.9.11-beta
       Используйте :py:class:`neuralprognoz.modelzoo.cudalstm.CudaLSTM` с ``statics_embedding``.

    Этот класс расширяет стандартный класс `CudaLSTM` для предварительной обработки статических входов с помощью embedding сети, prior
    перед объединением этих значений с динамическими (временными) входами. Используйте аргумент конфигурации `statics_embedding` для
    указания архитектуры fully-connected embedding сети. К выходам embedding сети не применяется функция активации.
    Для управления начальным смещением forget gate используйте аргумент конфигурации `initial_forget_bias`. Часто полезно установить
    это значение в положительное значение в начале обучения модели, чтобы держать forget gate закрытым и облегчить
    поток градиента.
    Класс `EmbCudaLSTM` поддерживает только предсказания на одном временном масштабе. Используйте `MTSLSTM` для обучения модели и получения
    предсказаний на нескольких временных масштабах одновременно.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # указать подмодули модели, которые могут быть использованы для дообучения. Имена должны совпадать с атрибутами класса
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(EmbCudaLSTM, self).__init__(cfg=cfg)

        warnings.warn('EmbCudaLSTM устарел, функциональность теперь является частью CudaLSTM.', FutureWarning)

        self.cudalstm = CudaLSTM(cfg)

        # дублировать некоторые члены, чтобы CustomLSTM мог получить к ним доступ
        self.embedding_net = self.cudalstm.embedding_net
        self.lstm = self.cudalstm.lstm
        self.head = self.cudalstm.head

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход модели EmbCudaLSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Выходы модели и промежуточные состояния в виде словаря.
                - `y_hat`: предсказания модели формы [batch size, sequence length, number of target variables].
                - `h_n`: скрытое состояние в последнем временном шаге последовательности формы [1, batch size, hidden size].
                - `c_n`: состояние ячейки в последнем временном шаге последовательности формы [1, batch size, hidden size].
        """

        return self.cudalstm.forward(data)
