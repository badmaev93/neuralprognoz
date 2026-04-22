import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuralprognoz.datautils.utils import get_frequency_factor, sort_frequencies
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.customlstm import _LSTMCell
from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)


class ODELSTM(BaseModel):
    """ODE-LSTM from [#]_.

    ODE-RNN пост-обрабатывает скрытое состояние обычного LSTM.
    Части этого кода заимствованы из https://github.com/mlech26l/learning-long-term-irregular-ts.
    
    Прямой проход в этой модели работает несколько иначе, чем в других моделях, поскольку ODE-LSTM полагается на
    выборки с нерегулярными интервалами. Чтобы смоделировать такую нерегулярность, мы агрегируем части входной последовательности до случайных
    частот. При этом мы стараемся не агрегировать слишком грубо непосредственно перед тем, как модель
    должна создать прогноз с высокой частотой.

    Поскольку такая агрегация означает, что части входной последовательности находятся на случайных частотах, мы не можем легко
    вернуть прогнозы для полной входной последовательности на каждой частоте. Вместо этого мы возвращаем только последовательности длины
    predict_last_n для каждой частоты (мы не применяем случайную агрегацию к этим последним временным шагам).

    Ниже описывается стратегия агрегации, реализованная в прямом проходе:

    1. срез один: шаги случайной частоты (cfg.ode_random_freq_lower_bound <= freq <= lowest-freq) до начала
                  второй по частоте входной последовательности.
    2. срез два: шаги случайной частоты (lowest-freq <= freq <= self._frequencies[1]) до начала
                  следующей по частоте входной последовательности.
    3. повторить срез два до начала самой высокой по частоте входной последовательности.
    4. срез три: шаги случайной частоты (self._frequencies[-2] <= freq <= highest-freq) до predict_last_n
                    самой низкой частоты.
    5. шаги самой низкой частоты для генерации predict_last_n прогнозов самой низкой частоты.
    6. повторить шаги четыре и пять для следующей по частоте (используя те же границы случайной частоты, но
       генерируя прогнозы для следующей по частоте).
       
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Ссылки
    ----------
    .. [#] Lechner, M.; Hasani, R.: Learning Long-Term Dependencies in Irregularly-Sampled Time Series. arXiv, 2020,
        https://arxiv.org/abs/2006.04418.
    """
    # Укажите подмодули модели, которые впоследствии можно будет использовать для тонкой настройки. Имена должны соответствовать атрибутам класса.
    module_parts = ['lstm_cell', 'ode_cell', 'head']

    def __init__(self, cfg: Config):
        super(ODELSTM, self).__init__(cfg=cfg)
        if len(cfg.use_frequencies) < 2:
            raise ValueError('ODELSTM требует как минимум две частоты.')
        if isinstance(cfg.dynamic_inputs, dict) or isinstance(cfg.hidden_size, dict):
            raise ValueError('ODELSTM не поддерживает входные переменные или размеры скрытого состояния для каждой частоты.')
        if isinstance(cfg.dynamic_inputs[0], list):
            raise ValueError('ODELSTM не поддерживает группы входных признаков.')

        # Обратите внимание, что frequency_factors и slice_timesteps имеют несколько иное значение здесь по сравнению с
        # MTSLSTM. Здесь frequency_factor относится к _самой низкой_ (а не к следующей по частоте) частоте.
        # slice_timesteps[freq] — это шаг входных данных (подсчитываемый в обратном порядке) на следующей по частоте
        # (более низкой) частоте, с которой доступны входные данные на частоте freq.
        self._frequency_factors = {}
        self._slice_timesteps = {}
        self._frequencies = sort_frequencies(cfg.use_frequencies)
        self._init_frequency_factors_and_slice_timesteps()

        # Начинаем подсчет количества входов
        self.input_size = len(cfg.dynamic_inputs + cfg.static_attributes + cfg.hydroatlas_attributes +
                              cfg.evolving_attributes)
        self._dynamic_inputs = cfg.dynamic_inputs

        if cfg.use_basin_id_encoding:
            self.input_size += cfg.number_of_basins
        if cfg.head.lower() == 'umal':
            self.input_size += 1

        self.lstm_cell = _LSTMCell(self.input_size, self.cfg.hidden_size, cfg.initial_forget_bias)
        self.ode_cell = _ODERNNCell(self.cfg.hidden_size,
                                    self.cfg.hidden_size,
                                    num_unfolds=self.cfg.ode_num_unfolds,
                                    method=self.cfg.ode_method)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.cfg.hidden_size, n_out=self.output_size)

    def _init_frequency_factors_and_slice_timesteps(self):
        for i, freq in enumerate(self._frequencies):
            frequency_factor = get_frequency_factor(self._frequencies[0], freq)
            if frequency_factor != int(frequency_factor):
                raise ValueError('Частоты должны быть кратны самой низкой частоте.')
            self._frequency_factors[freq] = int(frequency_factor)

            if i > 0:
                prev_frequency_factor = get_frequency_factor(self._frequencies[i - 1], freq)
                if self.cfg.predict_last_n[freq] % prev_frequency_factor != 0:
                    raise ValueError(
                        'На всех частотах predict_last_n должен совпадать с шагами следующей по частоте (более низкой) частоты.')

                if self.cfg.seq_length[freq] > self.cfg.seq_length[self._frequencies[i - 1]] * prev_frequency_factor:
                    raise ValueError('Частоты более высокого уровня должны иметь более короткие входные последовательности, чем частоты более низкого уровня.')

                # мы хотим передать состояние дня _перед_ началом следующей по частоте (более высокой) частоты,
                # потому что, например, среднее значение за день сохраняется в тот же день в 00:00.
                slice_timestep = self.cfg.seq_length[freq] / prev_frequency_factor
                if slice_timestep != int(slice_timestep):
                    raise ValueError('На всех частотах seq_length должен совпадать с шагами следующей по частоте (более низкой) частоты.')
                self._slice_timesteps[freq] = int(slice_timestep)

                # теоретически возможны следующие условия, но они усложнили бы реализацию
                # и, вероятно, редко бывают полезны.
                if self.cfg.predict_last_n[self._frequencies[
                        i - 1]] < self.cfg.predict_last_n[freq] / prev_frequency_factor:
                    raise NotImplementedError(
                        'Частоты более низкого уровня не могут иметь меньшие значения predict_last_n, чем частоты более высокого уровня.')

        if any(self.cfg.predict_last_n[f] / self._frequency_factors[f] > self._slice_timesteps[self._frequencies[-1]] /
               self._frequency_factors[self._frequencies[-2]] for f in self._frequencies):
            raise NotImplementedError('predict_last_n не может быть больше, чем длина последовательности самой высокой частоты.')

    def _prepare_inputs(self, data: Dict[str, torch.Tensor],
                        freq: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Объединяет все различные входы во входные данные временного ряда. """
        suffix = f"_{freq}"
        # транспонирование в [seq_length, batch_size, n_features]
        x_d = torch.cat([data[f'x_d{suffix}'][k] for k in self._dynamic_inputs], dim=-1).transpose(0, 1)

        # объединение всех входов
        if f'x_s{suffix}' in data and 'x_one_hot' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif f'x_s{suffix}' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        # добавить индикатор частоты. Он не будет использоваться как обычный вход, но ODE-RNN использует его для определения
        # прошедшего времени с момента последнего образца.
        frequency_factor = 1 / self._frequency_factors[freq]
        frequency_encoding = torch.ones(x_d.shape[0], x_d.shape[1], 1).to(x_d) * frequency_factor
        return torch.cat([x_d, frequency_encoding], dim=-1)

    def _randomize_freq(self, x_d: torch.Tensor, low_frequency: str, high_frequency: str) -> torch.Tensor:
        """Рандомизировать частоту входной последовательности """
        frequency_factor = int(get_frequency_factor(low_frequency, high_frequency))
        possible_aggregate_steps = list(filter(lambda n: frequency_factor % n == 0, range(1, frequency_factor + 1)))

        t = 0
        max_t = x_d.shape[0] / frequency_factor
        x_d_randomized = []
        while t < max_t:
            highfreq_slice = x_d[t * frequency_factor:(t + 1) * frequency_factor]

            # аггрегировать в рандомную частоту м-ду low и high
            random_aggregate_steps = np.random.choice(possible_aggregate_steps)
            if highfreq_slice.shape[0] % random_aggregate_steps == 0:
                randfreq_slice = highfreq_slice.view(-1, random_aggregate_steps, highfreq_slice.shape[1],
                                                     highfreq_slice.shape[2]).mean(dim=1)
                # обновить индикаторы частоты.
                randfreq_slice[:, :, -1] = random_aggregate_steps / self._frequency_factors[high_frequency]
            else:
                # не рандомизировать последний срез, если он не совпадает с шагами агрегации
                randfreq_slice = highfreq_slice
            x_d_randomized.append(randfreq_slice)

            t += 1

        return torch.cat(x_d_randomized, dim=0)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход на модели ODE-LSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Входные данные для прямого прохода. См. документацию по всем моделям для получения подробной информации о ключах словаря.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Предсказания модели для каждого целевого масштаба времени.
        """

        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        slice_one = self._randomize_freq(x_d[self._frequencies[0]][:-self._slice_timesteps[self._frequencies[1]]],
                                         self.cfg.ode_random_freq_lower_bound, self._frequencies[0])

        batch_size = slice_one.shape[1]
        h_0 = slice_one.data.new(batch_size, self.cfg.hidden_size).zero_()
        c_0 = slice_one.data.new(batch_size, self.cfg.hidden_size).zero_()
        h_n, c_n = self._run_odelstm(slice_one, h_0, c_0)

        prev_freq = self._frequencies[0]
        i = 1
        for freq in self._frequencies[1:-1]:
            to_randomize = x_d[freq][:-self._slice_timesteps[self._frequencies[i + 1]]]

            # slice_timestep текущей и следующей частоты идентичны, поэтому мы можем перейти к следующей частоте
            if len(to_randomize) == 0:
                continue
            # random-frequency шаги до начала входной последовательности самой высокой частоты.
            slice_two = self._randomize_freq(to_randomize, prev_freq, freq)
            h_n, c_n = self._run_odelstm(slice_two, h_n[:, -1], c_n)
            prev_freq = freq
            i += 1

        pred = {}
        prev_freq_end_step = 0
        for freq in self._frequencies:
            # запустить random-frequency шаги до predict_last_n этой частоты
            end_step = -int(self.cfg.predict_last_n[freq] * self._frequency_factors[self._frequencies[-1]] /
                            self._frequency_factors[freq])
            if end_step == 0:
                continue  # это означает, что текущая частота не должна быть предсказана, поэтому мы пропускаем ее.

            # проверить, что срез не будет пустым. Если он не будет, это означает, что predict_last_n предыдущей частоты
            # не совпадает с predict_last_n этой частоты, поэтому мы можем запустить random-frequency
            # входы до начала predict_last_n этой частоты.
            if end_step != prev_freq_end_step and end_step > -self.cfg.seq_length[self._frequencies[-1]]:
                slice_three = self._randomize_freq(x_d[self._frequencies[-1]][prev_freq_end_step:end_step], prev_freq,
                                                   self._frequencies[-1])
                h_n, c_n = self._run_odelstm(slice_three, h_n[:, -1], c_n)
                prev_freq_end_step = end_step

            # запустить predict_last_n шаги на целевой частоте
            pred_slice = x_d[freq][-self.cfg.predict_last_n[freq]:]
            h_n_out, _ = self._run_odelstm(pred_slice, h_n[:, -1], c_n)
            pred[f'y_hat_{freq}'] = self.head(self.dropout(h_n_out))['y_hat']

        return pred

    def _run_odelstm(self, input_slice: torch.Tensor, h_0: torch.Tensor,
                     c_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Выполнить прямой проход на модели ODE-LSTM. """
        h_x = (h_0, c_0)

        h_n = []
        for x_t in input_slice:
            h_0, c_0 = h_x

            # отделяем frequency индикатор от входных переменных
            t_input = x_t[:, :-1]
            t_elapsed = x_t[0, -1]

            lstm_out = self.lstm_cell(x_t=t_input, h_0=h_0, c_0=c_0)
            ode_out = self.ode_cell(lstm_out['h_n'], h_0, t_elapsed)

            h_x = (ode_out, lstm_out['c_n'])
            h_n.append(ode_out)

        # складываем h в [batch size, sequence length, hidden size]
        return torch.stack(h_n, 1), h_x[1]


class _ODERNNCell(nn.Module):
    """Ячейка ODE-RNN (Адаптировано из https://github.com/mlech26l/learning-long-term-irregular-ts) [#]_. 
    
    Параметры
    ----------
    input_size : int
        Размерность входных данных
    hidden_size : int
        Размерность скрытого состояния ячейки
    num_unfolds : int
        Количество шагов, на которые разбивается каждый временной шаг для решения ОДУ.
    method : {'euler', 'heun', 'rk4'}
        Метод, используемый для решения ОДУ (метод Эйлера, метод Хейна или метод Рунге-Кутты 4)
    
    Ссылки
    ----------
    .. [#] Lechner, M.; Hasani, R.: Learning Long-Term Dependencies in Irregularly-Sampled Time Series. arXiv, 2020,
        https://arxiv.org/abs/2006.04418.
    """

    def __init__(self, input_size: int, hidden_size: int, num_unfolds: int, method: str):
        super(_ODERNNCell, self).__init__()
        self.method = {
            'euler': self._euler,
            'heun': self._heun,
            'rk4': self._rk4,
        }[method]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_unfolds = num_unfolds

        self.w_ih = nn.Parameter(torch.FloatTensor(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_size))
        self.scale = nn.Parameter(torch.FloatTensor(hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        """Сброс параметров ячейки ODERNN. """
        nn.init.orthogonal_(self.w_hh)
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.scale, 1.0)

    def forward(self, new_hidden_state: torch.Tensor, old_hidden_state: torch.Tensor, elapsed: float) -> torch.Tensor:
        """Выполнить прямой проход на ячейке ODE-RNN.
        
        Параметры
        ----------
        new_hidden_state : torch.Tensor
            Текущее скрытое состояние, которое будет обновлено ячейкой ODE-RNN.
        old_hidden_state : torch.Tensor
            Предыдущее скрытое состояние.
        elapsed : float
            Прошедшее время между новым и старым скрытым состоянием.

        Возвращает
        -------
        torch.Tensor
            Предсказанное новое скрытое состояние
        """
        delta_t = elapsed / self.num_unfolds

        hidden_state = old_hidden_state
        for i in range(self.num_unfolds):
            hidden_state = self.method(new_hidden_state, hidden_state, delta_t)
        return hidden_state

    def _dfdt(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        h_in = torch.matmul(inputs, self.w_ih)
        h_rec = torch.matmul(hidden_state, self.w_hh)
        dh_in = self.scale * torch.tanh(h_in + h_rec + self.bias)
        dh = dh_in - hidden_state
        return dh

    def _euler(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        dy = self._dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def _heun(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        k1 = self._dfdt(inputs, hidden_state)
        k2 = self._dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def _rk4(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        k1 = self._dfdt(inputs, hidden_state)
        k2 = self._dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self._dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self._dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
