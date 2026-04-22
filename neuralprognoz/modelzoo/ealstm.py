from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class EALSTM(BaseModel):
    """Модель Entity-Aware LSTM (EA-LSTM).

    Эта модель была предложена Kratzert et al. [#]_ как вариант стандартного LSTM. Основное отличие заключается в том,
    что входные ворота EA-LSTM модулируются только с использованием статических входов, в то время как динамические 
    (временные ряды) входы используются во всех остальных частях модели (т.е. в воротах забывания, обновления ячейки и 
    выходных воротах).
    Чтобы контролировать начальное смещение ворот забывания, используйте аргумент конфигурации `initial_forget_bias`. 
    Часто полезно установить это значение в положительное значение в начале обучения модели, чтобы сохранить ворота 
    забывания закрытыми и облегчить поток градиента. 
    Класс `EALSTM` поддерживает только прогнозы в одном временном масштабе. Используйте `MTSLSTM` для обучения модели 
    на основе LSTM и получения прогнозов на нескольких временных разрешениях одновременно.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
        
    Ссылки
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning 
        universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, 
        Hydrol. Earth Syst. Sci., 23, 5089-5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.
    """
    # указать подмодули модели, которые могут быть использованы для дообучения. Имена должны совпадать с атрибутами класса
    module_parts = ['embedding_net', 'input_gate', 'dynamic_gates', 'head']

    def __init__(self, cfg: Config):
        super(EALSTM, self).__init__(cfg=cfg)
        self._hidden_size = cfg.hidden_size

        self.embedding_net = InputLayer(cfg)

        self.input_gate = nn.Linear(self.embedding_net.statics_output_size, cfg.hidden_size)

        # создать тензоры обучаемых параметров
        self.dynamic_gates = _DynamicGates(cfg=cfg, input_size=self.embedding_net.dynamics_output_size)
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def _cell(self, x: torch.Tensor, i: torch.Tensor, states: Tuple[torch.Tensor,
                                                                    torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single time step logic of EA-LSTM cell"""
        h_0, c_0 = states

        # calculate gates
        gates = self.dynamic_gates(h_0, x)
        f, o, g = gates.chunk(3, 1)

        c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)

        return h_1, c_1

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход модели EA-LSTM.

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
                    [batch size, sequence length, number of target variables].
                - `c_n`: состояние ячейки на последнем временном шаге последовательности размерностью 
                    [batch size, sequence length, number of target variables].
        """
        # возможно, пропустить динамические и статические входы через embedding слои
        x_d, x_s = self.embedding_net(data, concatenate_output=False)
        if x_s is None:
            raise ValueError('Необходимо передать x_s или x_one_hot в прямом проходе.')

        # TODO: переместить инициализацию скрытого и ячеечного состояний в init и сбрасывать состояния в прямом проходе в ноль.
        h_t = x_d.data.new(x_d.shape[1], self._hidden_size).zero_()
        c_t = x_d.data.new(x_d.shape[1], self._hidden_size).zero_()

        # пустые списки для временного хранения всех промежуточных скрытых/ячеечных состояний
        h_n, c_n = [], []

        # рассчитать входные ворота один раз, так как входы статичны
        i = torch.sigmoid(self.input_gate(x_s))

        # выполнить прямые шаги по входной последовательности
        for x_dt in x_d:

            h_t, c_t = self._cell(x_dt, i, (h_t, c_t))

            # сохранить промежуточное скрытое/ячеечное состояние в списке
            h_n.append(h_t)
            c_n.append(c_t)

        h_n = torch.stack(h_n, 0).transpose(0, 1)
        c_n = torch.stack(c_n, 0).transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(h_n)))

        return pred


class _DynamicGates(nn.Module):
    """Внутренний класс для объединения параметров динамических ворот в выделенный модуль PyTorch"""

    def __init__(self, cfg: Config, input_size: int):
        super(_DynamicGates, self).__init__()
        self.cfg = cfg
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * cfg.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(cfg.hidden_size, 3 * cfg.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * cfg.hidden_size))

        # инициализировать параметры
        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.cfg.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)

        if self.cfg.initial_forget_bias is not None:
            self.bias.data[:self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, h: torch.Tensor, x_d: torch.Tensor):
        gates = h @ self.weight_hh + x_d @ self.weight_ih + self.bias
        return gates
