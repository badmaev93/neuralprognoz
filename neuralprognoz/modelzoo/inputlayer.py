import itertools
import logging
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.fc import FC
from neuralprognoz.modelzoo.positional_encoding import PositionalEncoding
from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)

_EMBEDDING_TYPES = ['full_model', 'hindcast', 'forecast']

class InputLayer(nn.Module):
    """Входной слой для предварительной обработки статических и динамических входов.

    Этот модуль предоставляет необязательное встраивание (embedding) динамических и статических входов. Если ``dynamic_embeddings`` или
    ``static_embeddings`` указаны в виде словарей в конфигурации, то перед временной моделью будет вставлена полносвязная сеть встраивания.
    Словари имеют следующие ключи:

    - ``type`` (default 'fc'): Тип сети встраивания. В настоящее время поддерживается только 'fc' для полносвязной сети.
    - ``hiddens``: Список целых чисел, определяющих количество нейронов в каждом слое полносвязной сети.
      Последнее число — это количество выходных нейронов. Должен иметь длину не менее одного.
    - ``activation`` (default 'tanh'): функция активации сети. Поддерживаемые значения: 'tanh', 'sigmoid',
      'linear'. Функция активации не применяется к выходным нейронам, которые всегда имеют линейную активацию
      функция. Функция активации для выходных нейронов должна быть применена в основном классе модели.
    - ``dropout`` (default 0.0): Коэффициент отсечения (Dropout rate) применяется к сети встраивания.

    Примечание: этот модуль не поддерживает многочастотные запуски.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска
    """

    def __init__(self, cfg: Config, embedding_type: str = 'full_model'):
        super(InputLayer, self).__init__()

        self.embedding_type = embedding_type
        if embedding_type == 'full_model':
            dynamic_inputs = cfg.dynamic_inputs
            self._x_d_key = 'x_d'
        elif embedding_type == 'forecast':
            dynamic_inputs = cfg.forecast_inputs
            self._x_d_key = 'x_d_forecast'
        elif embedding_type == 'hindcast':
            dynamic_inputs = cfg.hindcast_inputs
            self._x_d_key = 'x_d_hindcast'
        else:
            raise ValueError(
                f'Тип встраивания {embedding_type} не распознан. '
                f'Должен быть одним из: {_EMBEDDING_TYPES}.'
            )
        if isinstance(dynamic_inputs, dict):
            if self.nan_handling_method:
                raise ValueError('InputLayer не поддерживает методы обработки NaN с несколькими частотами.')
            frequencies = list(dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError('InputLayer поддерживает только одночастотные данные')
            dynamics_input_sizes = [len(dynamic_inputs[frequencies[0]])]
            self._dynamic_inputs = {k: [v] for k, v in dynamic_inputs.items()}
        else:
            if isinstance(dynamic_inputs[0], str):
                # Объединить все признаки в одну группу признаков.
                self._dynamic_inputs = [dynamic_inputs]
            else:
                self._dynamic_inputs = dynamic_inputs
            if cfg.timestep_counter:
                # Добавить счетчик временных шагов в каждую группу признаков.
                if self.embedding_type == 'hindcast':
                    self._dynamic_inputs = [group + ['hindcast_counter'] for group in self._dynamic_inputs]
                elif self.embedding_type == 'forecast':
                    self._dynamic_inputs += [group + ['forecast_counter'] for group in self._dynamic_inputs]
            self.nan_handling_method = cfg.nan_handling_method
            self.attention = None
            self._nan_fill_value = 0.0
            if self.nan_handling_method == 'input_replacing':
                # +1 для флага NaN.
                dynamics_input_sizes = [sum(len(group) + 1
                                            for group in self._dynamic_inputs) + cfg.nan_handling_pos_encoding_size]
            elif self.nan_handling_method in ['masked_mean', 'attention']:
                dynamics_input_sizes = [len(group) + (cfg.nan_handling_pos_encoding_size
                                                        if self.nan_handling_method != 'attention' else 0)
                                        for group in self._dynamic_inputs]
            else:
                dynamics_input_sizes = [len(group) for group in self._dynamic_inputs]

        if cfg.head.lower() == "umal":
            dynamics_input_sizes = [size + 1 for size in dynamics_input_sizes]
            if isinstance(self._dynamic_inputs, dict):
                self._dynamic_inputs = {k: v + ['_tau'] for k, v in self._dynamic_inputs.items()}
            else:
                self._dynamic_inputs = [group + ['_tau'] for group in self._dynamic_inputs]

        self._num_autoregression_inputs = 0
        if cfg.autoregressive_inputs:
            self._num_autoregression_inputs = len(cfg.autoregressive_inputs)

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.statics_embedding, self.statics_output_size = \
            self._get_embedding_net(cfg.statics_embedding, statics_input_size, 'statics')

        self._pos_enc = None
        if cfg.nan_handling_pos_encoding_size > 0:
            if not self.nan_handling_method:
                raise NotImplementedError('Кодирование позиции поддерживается только для методов обработки NaN.')
            self._pos_enc = PositionalEncoding(embedding_dim=cfg.nan_handling_pos_encoding_size,
                                              position_type='concatenate',
                                              dropout=0.0,
                                              max_len=cfg.seq_length)

        dynamics_embeddings = []
        dynamics_output_sizes = []
        for dynamics_input_size in dynamics_input_sizes:
            group_embedding, group_output_size = self._get_embedding_net(cfg.dynamics_embedding,
                                                                         dynamics_input_size,
                                                                         'dynamics')
            dynamics_embeddings.append(group_embedding)
            dynamics_output_sizes.append(group_output_size)
        self.dynamics_embeddings = nn.ModuleList(dynamics_embeddings)
        if not all(size == dynamics_output_sizes[0] for size in dynamics_output_sizes):
            raise ValueError('Все выходные размеры встраивания динамических данных должны быть одинаковыми.')
        # Все выходные размеры одинаковы, просто используем первый.
        self.dynamics_output_size = dynamics_output_sizes[0]
        if self.nan_handling_method == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=self.dynamics_output_size, num_heads=1)
            self.query_embedding, _ = \
                    self._get_embedding_net(cfg.dynamics_embedding,
                                            (self.statics_output_size + len(self._dynamic_inputs)
                                             + cfg.nan_handling_pos_encoding_size),
                                            'query')

        if cfg.statics_embedding is None:
            self.statics_embedding_p_dropout = 0.0  # если сеть не имеет dropout статических данных, мы рассматриваем это как ноль
        else:
            self.statics_embedding_p_dropout = cfg.statics_embedding['dropout']
        if cfg.dynamics_embedding is None:
            self.dynamics_embedding_p_dropout = 0.0  # если сеть не имеет dropout динамических данных, мы рассматриваем это как ноль
        else:
            self.dynamics_embedding_p_dropout = cfg.dynamics_embedding['dropout']

        self.output_size = self.dynamics_output_size + self.statics_output_size + self._num_autoregression_inputs
        self.cfg = cfg

    @staticmethod
    def _get_embedding_net(embedding_spec: Optional[dict], input_size: int, purpose: str) -> Tuple[nn.Module, int]:
        """Получить сеть встраивания в соответствии с переданными спецификациями.

        Если `embedding_spec` равен None, возвращаемая сеть встраивания будет тождественной функцией.

        Параметры
        ----------
        embedding_spec : Optional[dict]
            Спецификация сети встраивания из конфигурации запуска или None.
        input_size : int
            Размер входов в сеть встраивания.
        purpose : str
            Назначение сети встраивания, используется для сообщений об ошибках.

        Возвращает
        -------
        Tuple[nn.Module, int]
            Сеть встраивания и ее выходной размер.
        """
        if embedding_spec is None:
            return nn.Identity(), input_size

        if input_size == 0:
            raise ValueError(f'Невозможно создать слой встраивания {purpose} с входным размером 0')

        emb_type = embedding_spec['type'].lower()
        if emb_type != 'fc':
            raise ValueError(f'Тип встраивания {emb_type} для {purpose} не поддерживается.')

        hiddens = embedding_spec['hiddens']
        if len(hiddens) == 0:
            raise ValueError(f'Список скрытых размеров "hiddens" для {purpose} должен содержать хотя бы один элемент')

        dropout = embedding_spec['dropout']
        activation = embedding_spec['activation']

        emb_net = FC(input_size=input_size, hidden_sizes=hiddens, activation=activation, dropout=dropout)
        return emb_net, emb_net.output_size

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]], concatenate_output: bool = True) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Выполнить прямой проход на входном слое.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Входные данные.
        concatenate_output : bool, optional
            Если True (по умолчанию), прямой проход будет конкатенировать статические входы с каждым временным шагом динамических данных.
            Если False, прямой проход вернет кортеж из (динамических, статических) входов.

        Возвращает
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Если `concatenate_output` истинно, возвращается единый тензор. В противном случае возвращается кортеж из одного тензора динамических входов и одного тензора статических входов.
        """
        features = self._dynamic_inputs
        if isinstance(features, dict):
            features = features[list(features.keys())[0]]

        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            x_s = None

        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)

        if self.nan_handling_method == 'masked_mean':
            dynamics_out = self._masked_mean_embedding(data[self._x_d_key])
        elif self.nan_handling_method == 'attention':
            dynamics_out = self._attention(data[self._x_d_key], statics_embedding=statics_out)
        elif self.nan_handling_method == 'input_replacing':
            dynamics_out = self._input_replacing_embedding(data[self._x_d_key])
        else:
            # транспонирование в [seq_length, batch_size, n_features]
            x_d = torch.cat([data[self._x_d_key][k] for k in itertools.chain(*features)], dim=-1).transpose(0, 1)
            dynamics_out = self.dynamics_embeddings[0](x_d)

        if not concatenate_output:
            ret_val = dynamics_out, statics_out
        else:
            if statics_out is not None:
                statics_out = statics_out.unsqueeze(0).repeat(dynamics_out.shape[0], 1, 1)
                ret_val = torch.cat([dynamics_out, statics_out], dim=-1)
            else:
                ret_val = dynamics_out

            # Добавление авторегрессионных входов в конец выхода.
            # Авторегрессионные входы не проходят через слой встраивания. Это не работает с NaN.
            if self._num_autoregression_inputs:
                x_autoregressive = torch.cat([data[self._x_d_key][k]
                                              for k in self.cfg.autoregressive_inputs], dim=-1).transpose(0, 1)
                ret_val = torch.cat([ret_val, x_autoregressive], dim=-1)

        return ret_val

    def _attention(self, x_d: dict[str, torch.Tensor], statics_embedding: torch.Tensor | None) -> torch.Tensor:
        """Механизм внимания со статическими + позиционным кодированием в качестве запроса, группами признаков в качестве ключей и значений."""
        if statics_embedding is None:
            raise ValueError('Внимание требует статических признаков.')

        dynamics_out = []
        masks = []
        for idx, feature_group in enumerate(self._dynamic_inputs):
            # транспонирование в [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            # Установка NaN в ноль для предотвращения NaN градиентов. Нули будут проигнорированы маской внимания.
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)
        dynamics_out = torch.stack(dynamics_out, dim=0)
        # Убедитесь, что есть что вернуть, даже если все значения - NaN.
        dynamics_out = torch.where(torch.isnan(dynamics_out).all(dim=0, keepdim=True),
                                   self._nan_fill_value, dynamics_out)

        n_groups, seq_len, batch_size, embed_dim = dynamics_out.shape
        stacked_masks = torch.stack(masks, dim=0).view(n_groups, seq_len * batch_size, 1)
        # query: (seq_len, batch_size, embed_dim)
        query = statics_embedding.unsqueeze(0).repeat(seq_len, 1, 1)
        if self._pos_enc is not None:
            query = self._pos_enc(query)
        query = torch.cat([query, stacked_masks.squeeze(-1).permute(1, 0).view(seq_len, batch_size, n_groups)],dim=-1)
        query = self.query_embedding(query)
        query = query.unsqueeze(0).view(1, seq_len * batch_size, embed_dim)

        # stacked masks: (n_groups, seq_len * batch_size)
        stacked_masks = stacked_masks.squeeze(-1)
        # key, value: (n_groups, seq_len * batch_size, embed_dim)
        key = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        value = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        key = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, key)
        value = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, value)

        # attn_mask: (seq_len * batch_size, 1, n_groups)
        attn_mask = stacked_masks.permute(1, 0).unsqueeze(1)
        # attention_out: (1, seq_len * batch_size, embed_dim)
        attention_out, _ = self.attention(query, key, value, attn_mask=attn_mask, need_weights=False)
        return attention_out.view(1, seq_len, batch_size, embed_dim).squeeze(0)

    def _masked_mean_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Выполняет masked mean embedding на входных данных."""
        dynamics_out = []
        masks = []
        for idx, feature_group in enumerate(self._dynamic_inputs):
            # транспонирование в [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            if self._pos_enc is not None:
                x_d_group = self._pos_enc(x_d_group)
            # Установка NaN в ноль для предотвращения NaN градиентов.
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            # Установка нулей обратно в NaN, чтобы они были проигнорированы в среднем.
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)
        dynamics_out = torch.stack(dynamics_out, dim=0)
        # Убедитесь, что среднее работает, даже если все значения - NaN.
        dynamics_out = torch.where(torch.isnan(dynamics_out).all(dim=0, keepdim=True),
                                   self._nan_fill_value, dynamics_out)
        return torch.nanmean(dynamics_out, dim=0)

    def _input_replacing_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Добавляет маски входов к входам и устанавливает NaN в ноль."""
        dynamics = []
        for feature_group in self._dynamic_inputs:
            # транспонирование в [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            x_d_group = torch.where(mask, self._nan_fill_value, x_d_group)
            dynamics.append(x_d_group)
            dynamics.append(mask.to(torch.float32))
        dynamics = torch.cat(dynamics, dim=-1)
        if self._pos_enc is not None:
            dynamics = self._pos_enc(dynamics)
        return self.dynamics_embeddings[0](dynamics)

    def __getitem__(self, item: str) -> nn.Module:
        # требуется для доступа в виде словаря при замораживании градиентов подмодулей при дообучении
        if item == "statics_embedding":
            return self.statics_embedding
        elif item == "dynamics_embedding":
            return self.dynamics_embedding
        else:
            raise KeyError(f"Невозможно получить доступ к {item} в InputLayer")
