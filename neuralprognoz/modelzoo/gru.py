from typing import Dict

import torch
from torch import nn

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class GRU(BaseModel):
    """Класс Gated Recurrent Unit (GRU), основанный на реализации GRU в PyTorch.

    Этот класс реализует стандартный GRU в сочетании с моделью, как указано в конфигурации. Все признаки
    (временные ряды и статические) объединяются и передаются в GRU напрямую.
    Класс `GRU` поддерживает только прогнозы в одном временном масштабе.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # указать субмодули модели, которые позже можно будет использовать для дообучения (finetuning). Имена должны совпадать с атрибутами класса
    module_parts = ['embedding_net', 'gru', 'head']

    def __init__(self, cfg: Config):

        super(GRU, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.gru = nn.GRU(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход модели GRU.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь с выходами модели и промежуточными состояниями. 
                - `y_hat`: прогнозы модели размерностью [batch size, sequence length, number of target variables].
                - `h_n`: скрытое состояние на последнем временном шаге последовательности размерностью 
                    [batch size, 1, hidden size].
        """
        # возможно, пропустить динамические и статические входы через embedding слои, затем объединить их
        x_d = self.embedding_net(data, concatenate_output=True)

        # выполнить прямой проход GRU
        gru_output, h_n = self.gru(input=x_d)

        # изменить форму на [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)

        pred = {'h_n': h_n}

        # добавить финальный выход в словарь предсказаний
        # (this will contain the 'y_hat')
        pred.update(self.head(self.dropout(gru_output.transpose(0, 1))))

        return pred
