from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config


class CudaLSTM(BaseModel):
    """Класс модели LSTM, основанный на классе LSTM PyTorch CUDA.

    Этот класс реализует стандартный LSTM в сочетании с моделью, как указано в конфигурации. В зависимости от
    настроек встраивания, статические и/или динамические признаки могут или не могут быть переданы через сети встраивания перед тем, как быть объединенными и переданными через LSTM.
    Чтобы контролировать начальный наклон забывающих ворот, используйте аргумент конфигурации `initial_forget_bias`. Часто полезно установить
    это значение в положительное значение в начале обучения модели, чтобы держать забывающие ворота закрытыми и облегчить
    поток градиента.
    Класс `CudaLSTM` поддерживает только прогнозы в одном временном масштабе. Используйте `MTSLSTM` для обучения модели и получения
    прогнозов на нескольких временных разрешениях одновременно.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # укажите подмодули модели, которые позже можно будет использовать для дообучения (finetuning).
    # Имена должны совпадать с атрибутами класса
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(CudaLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Особая инициализация некоторых весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход через модель CudaLSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий выходные данные модели и промежуточные состояния.
                - `y_hat`: прогнозы модели формы [batch size, sequence length, number of target variables].
                - `h_n`: скрытое состояние в последнем временном шаге последовательности формы [batch size, 1, hidden size].
                - `c_n`: состояние ячейки в последнем временном шаге последовательности формы [batch size, 1, hidden size].
        """
        # возможно, передать динамические и статические входы через слои встраивания, затем объединить их
        x_d = self.embedding_net(data)
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # изменить форму на [batch_size, seq, n_hiddens]
        lstm_output = lstm_output.transpose(0, 1)
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
