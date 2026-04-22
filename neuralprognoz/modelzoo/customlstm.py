import math
from collections import defaultdict

from typing import Dict, Union

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.cudalstm import CudaLSTM
from neuralprognoz.modelzoo.embcudalstm import EmbCudaLSTM
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class CustomLSTM(BaseModel):
    """Пользовательская реализация LSTM с поддержкой выделенной сети встраивания.

    Идея этой модели заключается в том, чтобы в основном использоваться в качестве аналитического инструмента, где вы можете обучить модель, используя оптимизированные
    классы `CudaLSTM` или `EmbCudaLSTM`, а затем скопировать веса в эту модель для более глубокого анализа сети
    (например, проверки состояний модели или активаций ворот). Преимущество этой реализации заключается в том, что она возвращает
    всю временную серию векторов состояний и активаций ворот.
    Однако вы также можете использовать этот класс модели для обучения, но обратите внимание, что он будет значительно медленнее, чем его
    оптимизированные аналоги. В зависимости от настроек встраивания, статические и/или динамические признаки могут или не могут быть переданы
    через сети встраивания перед тем, как быть объединенными и переданными через LSTM.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Пример
    -------
    >>> # Пример копирования весов оптимизированного `CudaLSTM` или `EmbCudaLSTM` в экземпляр `CustomLSTM`
    >>> cfg = ... # Экземпляр конфигурации, соответствующий исходной, оптимизированной модели
    >>> optimized_lstm = ... # Экземпляр модели `CudaLSTM` или `EmbCudaLSTM`
    >>>
    >>> # Используйте исходную конфигурацию для инициализации этой модели, чтобы различать `CudaLSTM` и `EmbCudaLSTM`
    >>> custom_lstm = CustomLSTM(cfg=cfg)
    >>>
    >>> # Скопируйте веса в экземпляр `LSTM`.
    >>> custom_lstm.copy_weights(optimized_lstm)
    """

    def __init__(self, cfg: Config):
        super(CustomLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self._hidden_size = cfg.hidden_size

        self.cell = _LSTMCell(input_size=self.embedding_net.output_size,
                              hidden_size=self._hidden_size,
                              initial_forget_bias=cfg.initial_forget_bias)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self._hidden_size, n_out=self.output_size)

    def forward(self,
                data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход через модель LSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.
        h_0 : torch.Tensor, optional
            Начальное скрытое состояние, по умолчанию 0.
        c_0 : torch.Tensor, optional
            Начальное состояние ячейки, по умолчанию 0.

        Returns
        -------
        Dict[str, torch.Tensor]
            Выходные данные модели и все промежуточные состояния и активации ворот в виде словаря.
        """
        # возможно, передать динамические и статические входные данные через слои встраивания, затем объединить их
        x_d = self.embedding_net(data, concatenate_output=True)

        seq_len, batch_size, _ = x_d.size()

        # TODO: переместить инициализацию скрытого и ячеечного состояний в init и сбрасывать состояния в прямом проходе до нуля.
        if h_0 is None:
            h_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        if c_0 is None:
            c_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        h_x = (h_0, c_0)

        output = defaultdict(list)
        for x_t in x_d:
            h_0, c_0 = h_x
            cell_output = self.cell(x_t=x_t, h_0=h_0, c_0=c_0)

            h_x = (cell_output['h_n'], cell_output['c_n'])

            for key, cell_out in cell_output.items():
                output[key].append(cell_out)

        # стек в [batch size, sequence length, hidden size]
        pred = {key: torch.stack(val, 1) for key, val in output.items()}
        pred.update(self.head(self.dropout(pred['h_n'])))
        return pred

    def copy_weights(self, optimized_lstm: Union[CudaLSTM, EmbCudaLSTM]):
        """Копирует веса из `CudaLSTM` или `EmbCudaLSTM` в этот класс модели

        Параметры
        ----------
        optimized_lstm : Union[CudaLSTM, EmbCudaLSTM]
            Экземпляр модели `CudaLSTM` (neuralprognoz.modelzoo.cudalstm) или `EmbCudaLSTM`
            (neuralprognoz.modelzoo.embcudalstm).
            
        Вызывает
        ------
        RuntimeError
            Если `optimized_lstm` является `EmbCudaLSTM`, но этот экземпляр модели не был создан с сетью встраивания.
        """

        assert isinstance(optimized_lstm, CudaLSTM) or isinstance(optimized_lstm, EmbCudaLSTM)

        self.embedding_net.load_state_dict(optimized_lstm.embedding_net.state_dict())

        # копирование весов ячейки lstm
        self.cell.copy_weights(optimized_lstm.lstm, layer=0)

        # копирование весов головы
        self.head.load_state_dict(optimized_lstm.head.state_dict())


class _LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, initial_forget_bias: float = 0.0):
        super(_LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias

        self.w_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size, hidden_size))
        self.w_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size, input_size))

        self.b_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.b_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        stdv = math.sqrt(3 / self.hidden_size)
        for weight in self.parameters():
            if len(weight.shape) > 1:
                weight.data.uniform_(-stdv, stdv)
            else:
                nn.init.zeros_(weight)

        if self.initial_forget_bias != 0:
            self.b_hh.data[self.hidden_size:2 * self.hidden_size] = self.initial_forget_bias

    def forward(self, x_t: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor) -> Dict[str, torch.Tensor]:
        gates = h_0 @ self.w_hh.T + self.b_hh + x_t @ self.w_ih.T + self.b_ih
        i, f, g, o = gates.chunk(4, 1)

        c_1 = c_0 * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)

        return {'h_n': h_1, 'c_n': c_1, 'i': i, 'f': f, 'g': g, 'o': o}

    def copy_weights(self, cudnnlstm: nn.Module, layer: int):

        assert self.hidden_size == cudnnlstm.hidden_size
        assert self.input_size == cudnnlstm.input_size

        self.w_hh.data = getattr(cudnnlstm, f"weight_hh_l{layer}").data
        self.w_ih.data = getattr(cudnnlstm, f"weight_ih_l{layer}").data
        self.b_hh.data = getattr(cudnnlstm, f"bias_hh_l{layer}").data
        self.b_ih.data = getattr(cudnnlstm, f"bias_ih_l{layer}").data
