import logging
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from neuralprognoz.datautils.utils import get_frequency_factor, sort_frequencies
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)


class MTSLSTM(BaseModel):
    """Multi-Timescale LSTM (MTS-LSTM) from Gauch et al. [#]_.

    Архитектура LSTM, которая позволяет одновременно прогнозировать на нескольких временных масштабах в одной модели.
    Существует два варианта этой модели: MTS-LTSM и sMTS-LSTM (общий MTS-LSTM). MTS-LSTM обрабатывает входные данные на
    низких временных разрешениях до определенного момента времени. Затем LSTM разделяется на одну ветвь для каждого целевого временного масштаба.
    Каждая ветвь обрабатывает входные данные на своем соответствующем временном масштабе. Наконец, один блок прогнозирования для каждого временного масштаба генерирует
    прогнозы для этого временного масштаба на основе выхода LSTM.
    Опционально можно указать:
    - разный hidden size для каждой ветви LSTM (используйте словарь в аргументе конфигурации ``hidden_size``)
    - разные динамические входные переменные для каждого временного масштаба (используйте словарь в аргументе конфигурации ``dynamic_inputs``)
    - стратегию передачи состояний от начального общего LSTM низкого разрешения к LSTM более высокого разрешения для каждого временного масштаба. По умолчанию используется линейный передаточный слой, но вы можете указать 'identity', чтобы использовать тождественную операцию, или 'None', чтобы отключить любую передачу (через аргумент конфигурации ``transfer_mtlstm_states``).
    - сети встраивания для статических и динамических входов с использованием аргументов конфигурации ``statics_embedding`` и ``dynamics_embedding``

    Вариант sMTS-LSTM имеет ту же общую архитектуру, но веса ветвей для каждого временного масштаба (включая выходные блоки) разделяются.
    Таким образом, в отличие от MTS-LSTM, sMTS-LSTM не может использовать скрытые размеры для каждого временного масштаба или динамические входные переменные.
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Ссылки
    ----------
    .. [#] Gauch, M., Kratzert, F., Klotz, D., Grey, N., Lin, J., and Hochreiter, S.: Rainfall-Runoff Prediction at
        Multiple Timescales with a Single Long Short-Term Memory Network, arXiv Preprint,
        https://arxiv.org/abs/2010.07921, 2020.
    """
    # укажите подмодули модели, которые впоследствии можно будет использовать для тонкой настройки. Имена должны соответствовать атрибутам класса.
    module_parts = ['embedding_net', 'lstms', 'transfer_fcs', 'heads']

    def __init__(self, cfg: Config):
        super(MTSLSTM, self).__init__(cfg=cfg)
        self.lstms = None
        self.transfer_fcs = None
        self.heads = None
        self.dropout = None

        self._slice_timestep = {}
        self._frequency_factors = []

        self._seq_lengths = cfg.seq_length
        self._is_shared_mtslstm = self.cfg.shared_mtslstm  # по умолчанию: отдельный LSTM для каждого временного масштаба
        self._transfer_mtslstm_states = self.cfg.transfer_mtslstm_states  # по умолчанию: линейный передаточный слой
        transfer_modes = [None, "None", "identity", "linear"]
        if self._transfer_mtslstm_states["h"] not in transfer_modes \
                or self._transfer_mtslstm_states["c"] not in transfer_modes:
            raise ValueError(f"MTS-LSTM поддерживает режимы передачи состояний {transfer_modes}")

        if len(cfg.use_frequencies) < 2:
            raise ValueError("MTS-LSTM ожидает более одной входной частоты")
        self._frequencies = sort_frequencies(cfg.use_frequencies)

        # инициализация сетей встраивания для каждой частоты, если определены встраивания
        self.embedding_net = None
        if (cfg.dynamics_embedding is not None) or (cfg.statics_embedding is not None):
            self.embedding_net = nn.ModuleDict()
            for freq in self._frequencies:
                freq_params = {
                    'model': 'mtslstm',
                    'dynamic_inputs': cfg.dynamic_inputs[freq] if isinstance(cfg.dynamic_inputs, dict) else cfg.dynamic_inputs,
                    'dynamics_embedding': cfg.dynamics_embedding,
                    'static_attributes': cfg.static_attributes,
                    'statics_embedding': cfg.statics_embedding,
                    'use_basin_id_encoding': cfg.use_basin_id_encoding,
                    'number_of_basins': cfg.number_of_basins,
                    'head': cfg.head,
                    'seq_length': cfg.seq_length[freq] if isinstance(cfg.seq_length, dict) else cfg.seq_length
                }
                freq_cfg = Config(freq_params)
                self.embedding_net[freq] = InputLayer(freq_cfg, embedding_type='full_model')

        # начинаем считать количество входов
        input_sizes = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)

        # если is_shared_mtslstm, LSTM получает дополнительный флаг частоты в качестве входа.
        if self._is_shared_mtslstm:
            input_sizes += len(self._frequencies)

        if cfg.use_basin_id_encoding:
            input_sizes += cfg.number_of_basins
        if cfg.head.lower() == "umal":
            input_sizes += 1

        if isinstance(cfg.dynamic_inputs, list):
            if isinstance(cfg.dynamic_inputs[0], list):
                raise ValueError("MTS-LSTM не поддерживает группы входных признаков.")
            self._dynamic_inputs = {freq: cfg.dynamic_inputs for freq in self._frequencies}
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs) for freq in self._frequencies}
        else:
            self._dynamic_inputs = cfg.dynamic_inputs
            if self._is_shared_mtslstm:
                raise ValueError(f'Разные входы не допускаются, если используется shared_mtslstm.')
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs[freq]) for freq in self._frequencies}

        if not isinstance(cfg.hidden_size, dict):
            LOGGER.info("Не указан конкретный скрытый размер для частот. Для всех используется одинаковый скрытый размер.")
            self._hidden_size = {freq: cfg.hidden_size for freq in self._frequencies}
        else:
            self._hidden_size = cfg.hidden_size

        if (self._is_shared_mtslstm
            or self._transfer_mtslstm_states["h"] == "identity"
            or self._transfer_mtslstm_states["c"] == "identity") \
                and any(size != self._hidden_size[self._frequencies[0]] for size in self._hidden_size.values()):
            raise ValueError("Все скрытые размеры должны быть одинаковыми, если используется shared_mtslstm или state transfer=identity.")

        # создаем слой в зависимости от выбранных частот
        self._init_modules(input_sizes)
        self._reset_parameters()

        # частотные коэффициенты необходимы для определения временного шага передачи информации
        self._init_frequency_factors_and_slice_timesteps()

    def _init_modules(self, input_sizes: Dict[str, int]):
        self.lstms = nn.ModuleDict()
        self.transfer_fcs = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)
        for idx, freq in enumerate(self._frequencies):
            if self.embedding_net is not None:
                freq_input_size = self.embedding_net[freq].output_size
                if self._is_shared_mtslstm:
                    freq_input_size += len(self._frequencies)
                if self.cfg.head.lower() == "umal":
                    freq_input_size += 1
            else:
                freq_input_size = input_sizes[freq]

            if self._is_shared_mtslstm and idx > 0:
                self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]  # один и тот же LSTM для всех частот.
                self.heads[freq] = self.heads[self._frequencies[idx - 1]]  # один и тот же выходной блок для всех частот.
            else:
                self.lstms[freq] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
                self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)

            if idx < len(self._frequencies) - 1:
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Linear(self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]])
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity()
                    else:
                        pass

    def _init_frequency_factors_and_slice_timesteps(self):
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                frequency_factor = get_frequency_factor(freq, self._frequencies[idx + 1])
                if frequency_factor != int(frequency_factor):
                    raise ValueError('Adjacent frequencies must be multiples of each other.')
                self._frequency_factors.append(int(frequency_factor))
                # мы хотим передать состояние дня _перед_ началом следующей более высокой частоты,
                # потому что, например, среднее значение дня хранится в тот же день в 00:00 утра.
                slice_timestep = int(self._seq_lengths[self._frequencies[idx + 1]] / self._frequency_factors[idx])
                self._slice_timestep[freq] = slice_timestep

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            for freq in self._frequencies:
                hidden_size = self._hidden_size[freq]
                self.lstms[freq].bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias

    def _add_frequency_one_hot_encoding(self, x_d: torch.Tensor, freq: str) -> torch.Tensor:
        idx = self._frequencies.index(freq)
        one_hot_freq = torch.zeros(x_d.shape[0], x_d.shape[1], len(self._frequencies), device=x_d.device)
        one_hot_freq[:, :, idx] = 1
        return torch.cat([x_d, one_hot_freq], dim=2)

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str) -> torch.Tensor:
        """Объединяет все различные входы во вход временного ряда"""
        if self.embedding_net is not None:
            # использовать сеть встраивания, если доступна
            input_data = {
                'x_d': data[f'x_d_{freq}'],
                'x_s': data.get('x_s'),
                'x_one_hot': data.get('x_one_hot')
            }
            # если доступен конкретный x_s для частоты, использовать его
            if f'x_s_{freq}' in data:
                input_data['x_s'] = data[f'x_s_{freq}']
                
            # отфильтровать значения None
            input_data = {k: v for k, v in input_data.items() if v is not None}
            x_d = self.embedding_net[freq](input_data, concatenate_output=True)
                       
            # добавить one-hot кодирование частоты, если используется shared_mtslstm
            if self._is_shared_mtslstm:
                x_d = self._add_frequency_one_hot_encoding(x_d, freq)

        else:
            # использовать входные данные напрямую без встраивания, следуя оригинальной реализации
            suffix = f"_{freq}"
            
            # объединить все тензоры признаков из словаря
            feature_tensors = []
            for feature_name, feature_tensor in data[f'x_d{suffix}'].items():
                feature_tensors.append(feature_tensor)
            
            if feature_tensors:
                # объединить все признаки по измерению признаков
                x_d = torch.cat(feature_tensors, dim=-1).transpose(0, 1)
            else:
                # не найдены динамические признаки, что недопустимо
                raise ValueError(f"Не найдены динамические признаки для частоты {freq}.")

            # объединить все статические и one-hot закодированные признаки
            if f'x_s{suffix}' in data and 'x_one_hot' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
            elif f'x_s{suffix}' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_s' in data:
                x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_one_hot' in data:
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_one_hot], dim=-1)
            else:
                pass

            if self._is_shared_mtslstm:
                x_d = self._add_frequency_one_hot_encoding(x_d, freq)
        
        return x_d

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход на модели MTS-LSTM.
        
        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Входные данные для прямого прохода. См. обзорную документацию всех моделей для получения подробной информации о ключах словаря.

        Returns
        -------
        Dict[str, torch.Tensor]
            Прогнозы модели для каждого целевого временного масштаба.
        """
        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        # начальные состояния для самых низких частот устанавливаются в нули
        batch_size = x_d[self._frequencies[0]].shape[1]
        lowest_freq_hidden_size = self._hidden_size[self._frequencies[0]]
        h_0_transfer = x_d[self._frequencies[0]].new_zeros((1, batch_size, lowest_freq_hidden_size))
        c_0_transfer = torch.zeros_like(h_0_transfer)

        outputs = {}
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                # получаем прогнозы и состояние до временного шага передачи информации
                slice_timestep = self._slice_timestep[freq]
                lstm_output_slice1, (h_n_slice1, c_n_slice1) = self.lstms[freq](x_d[freq][:-slice_timestep],
                                                                                (h_0_transfer, c_0_transfer))

                # проецируем состояния через скрытый слой в размерности следующего LSTM
                if self._transfer_mtslstm_states["h"] is not None:
                    h_0_transfer = self.transfer_fcs[f"h_{freq}"](h_n_slice1)
                if self._transfer_mtslstm_states["c"] is not None:
                    c_0_transfer = self.transfer_fcs[f"c_{freq}"](c_n_slice1)

                # получаем прогнозы оставшейся части и объединяем результаты
                lstm_output_slice2, _ = self.lstms[freq](x_d[freq][-slice_timestep:], (h_n_slice1, c_n_slice1))
                lstm_output = torch.cat([lstm_output_slice1, lstm_output_slice2], dim=0)

            else:
                # для самой высокой частоты мы можем передать всю последовательность сразу
                lstm_output, _ = self.lstms[freq](x_d[freq], (h_0_transfer, c_0_transfer))

            head_out = self.heads[freq](self.dropout(lstm_output.transpose(0, 1)))
            outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})

        return outputs
