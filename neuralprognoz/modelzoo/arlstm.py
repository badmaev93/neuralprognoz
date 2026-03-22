import math
from collections import defaultdict

import re
from typing import Dict, List

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class ARLSTM(BaseModel):
    """Авторегрессионная LSTM-сеть.

    Эта модель предполагает, что *последний* элемент динамического входного сигнала (x_d) — это наблюдение, которое должно соответствовать
    целевым данным, отстающим на целое число временных шагов. Если эти данные отсутствуют (NaN), они заменяются
    на прогноз модели с тем же отставанием. Модель добавляет дополнительный динамический входной сигнал, который служит двоичным
    флагом, указывающим, получен ли авторегрессионный входной сигнал на конкретном временном шаге в результате наблюдения или моделирования.
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """

    def __init__(self, cfg: Config):
        super(ARLSTM, self).__init__(cfg=cfg)

        self._ar_shift = self._get_ar_shift()
        self._num_ar_inputs = len(self.cfg.autoregressive_inputs)
        if self.output_size != self._num_ar_inputs:
            raise ValueError('В настоящее время AR-LSTM работает только в том случае, если все выходы используются как лаггированные входы.')

        self.embedding_net = InputLayer(cfg)

        # увеличить размер входных данных для двоичного флага
        input_size = self.embedding_net.output_size + self._num_ar_inputs
        self.cell = nn.LSTM(input_size=input_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _get_ar_shift(self) -> int:
        shifts = set()
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Авторегрессионные входные данные должны быть сдвинутой переменной в форме <variable>_shift<lag> ',
                                f'где <lag> должен быть integer. Вместо этого получил: {input}.')
            if not capture[1] in self.cfg.target_variables:
                raise ValueError('Авторегрессионные входные данные должны быть смещенной целевой переменной ',
                                f'Вместо этого получил смещенную версию: {capture[1]}.')
            shifts.add(int(capture[2]))
        if len(shifts) > 1:
            raise ValueError('В настоящее время допускается только один авторегрессионный сдвиг. Все входные данные авторегрессии должны использовать один и тот же сдвиг.')
        shift = shifts.pop()
        if shift <= 0:
            raise ValueError('Авторегрессионные входные данные должны быть сдвинуты как минимум на один временной шаг.')
        return shift    
    
    def _reset_parameters(self):
        """Специальная инициализация некоторых весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.cell.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self,
                data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход по модели авторегрессии LSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные характеристики как пары "ключ-значение".

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Выходные данные и промежуточные состояния модели можно представить в виде словаря.
                - `y_hat`: модельные прогнозы формы [batch size, sequence length, number of target variables].
                - `lstm_output`: полный временной ряд скрытых состояний из lstm [batch size, sequence length, hidden size].
                - `h_n`: скрытое состояние на последнем временном шаге последовательности формы [batch size, 1, hidden size].
                - `c_n`: состояние ячейки на последнем временном шаге последовательности формы [batch size, 1, hidden size].
        """
        # возможно, передавать динамические и статические входные данные через встроенные слои, а затем объединять их
        x_d = self.embedding_net(data)

        _, batch_size, _ = x_d.size()

        if h_0 is None:
            h_0 = x_d.new_zeros((1, batch_size, self.cfg.hidden_size))
        if c_0 is None:
            c_0 = x_d.new_zeros((1, batch_size, self.cfg.hidden_size))

        # инициализировать флаги для авторегрессионных входов - все флаги начинаются с 0, что указывает на obs
        # флаги добавляются как последний вход
        ar_flags = x_d.new_zeros((batch_size, self._num_ar_inputs))
        
        # инициализировать "последнее" предсказание для замены любых отсутствующих значений AR на первых временных шагах
        last_prediction = x_d.new_zeros((self._ar_shift, batch_size, self.output_size))

        # инициализировать словарь для сохранения выходных данных
        lstm_output = []
        y_hat = []

        # вручную перебирать временные шаги
        for x_t in x_d:

            # найти места с пропущенными данными (NaN) и заменить их последними прогнозами
            x_embd = x_t[:, :-self._num_ar_inputs]
            x_ar = x_t[:, -self._num_ar_inputs:].clone()
            replace_indexes = torch.isnan(x_ar)
            x_ar[replace_indexes] = last_prediction[-1, replace_indexes]
            ar_flags[:, :] = 0
            ar_flags[replace_indexes] = 1

            # один временной шаг LSTM
            cell_inputs = torch.unsqueeze(torch.concat([x_embd, x_ar, ar_flags], -1), 0)
            cell_output, (h_0, c_0) = self.cell(cell_inputs, (h_0, c_0))

            # добавить все выходные данные временных шагов в выходной словарь
            lstm_output.append(cell_output.transpose(0, 1))

            # сохранить последний прогноз
            last_prediction[1:] = last_prediction[:-1,].clone()
            prediction = torch.squeeze(self.head(self.dropout(h_0.transpose(0, 1)))['y_hat'], dim=1)
            last_prediction[0] = prediction
            y_hat.append(prediction)

        # сложить все выходные данные в соответствии с размерами в функциональном документе
        pred = {
            'lstm_output': torch.concat(lstm_output, 1), 
            'h_n': h_0.transpose(0, 1), 
            'c_n': c_0.transpose(0, 1),
            'y_hat': torch.stack(y_hat, 1),
        }
        return pred
