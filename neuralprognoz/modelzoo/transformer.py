import logging
import math
from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.positional_encoding import PositionalEncoding
from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)


class Transformer(BaseModel):
    """Класс модели Трансформер, который полагается на класс TransformerEncoder из PyTorch.

    Этот класс реализует кодировщик сети трансформера, который может быть использован для регрессии.
    Если количество входов не делится на количество голов трансформера (``transformer_nheads``), необходимо
    использовать сеть встраивания, которая это гарантирует. Для этого используйте ``statics/dynamics_embedding``,
    так что статические/динамические входы будут проходить через сети встраивания перед конкатенацией. Сеть встраивания
    затем отобразит статические и динамические признаки в размер ``statics/dynamics_embedding['hiddens'][-1]``, так что
    общий размер встраивания будет суммой этих значений.
    Конфигурация модели указывается в файле конфигурации с использованием следующих параметров:

    * ``transformer_positional_encoding_type``: варианты "sum" или "concatenate" позиционного кодирования к другим
      входам модели.
    * ``transformer_positional_dropout``: коэффициент отсева, применяемый к позиционному кодированию.
    * ``seq_length``: целое число временных шагов для обработки во входной последовательности.
    * ``transformer_nheads``: количество голов самовнимания.
    * ``transformer_dim_feedforward``: размерность полносвязных сетей между головами самовнимания.
    * ``transformer_dropout``: коэффициент отсева в полносвязных сетях между головами самовнимания.
    * ``transformer_nlayers``: количество слоев самовнимания + полносвязных слоев.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # определить субмодули модели, которые могут быть использованы для дообучения. Имена должны совпадать с атрибутами класса
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(Transformer, self).__init__(cfg=cfg)

        # сеть встраивания перед трансформером
        self.embedding_net = InputLayer(cfg)

        # убедиться, что количество входов в слой самовнимания делится на количество голов
        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Размер встраивания должен делиться на количество голов трансформера. "
                             "Используйте statics_embedding/dynamics_embedding и embedding_hiddens для указания встраивания.")

        self._sqrt_embedding_dim = math.sqrt(self.embedding_net.output_size)

        # позиционный кодировщик
        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Нераспознанный тип позиционного кодирования: {self._positional_encoding_type}")
        self.positional_encoder = PositionalEncoding(embedding_dim=self.embedding_net.output_size,
                                                      dropout=cfg.transformer_positional_dropout,
                                                      position_type=cfg.transformer_positional_encoding_type,
                                                      max_len=cfg.seq_length)

        # позиционная маска
        self._mask = None

        # энкодер
        encoder_layers = nn.TransformerEncoderLayer(d_model=encoder_dim,
                                                    nhead=cfg.transformer_nheads,
                                                    dim_feedforward=cfg.transformer_dim_feedforward,
                                                    dropout=cfg.transformer_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None)

        # выходной слой (вместо декодера)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)

        # инициализация весов и смещений
        self._reset_parameters()

    def _reset_parameters(self):
        # эта стратегия инициализации была протестирована эмпирически, но может не быть универсально лучшей стратегией во всех случаях.
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Прямой проход через модель Трансформер без декодера.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Выходные данные модели и промежуточные состояния в виде словаря.
                - `y_hat`: предсказания модели формы [размер батча, число временных шагов, число целевых переменных].
        """
        # пропустить динамические и статические входы через слои встраивания, затем объединить их
        x_d = self.embedding_net(data)

        positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim)

        # маскировать будущие значения
        if self._mask is None or self._mask.size(0) != len(x_d):
            self._mask = torch.triu(x_d.new_full((len(x_d), len(x_d)), fill_value=float('-inf')), diagonal=1)

        # кодирование
        output = self.encoder(positional_encoding, self._mask)

        # выходной слой
        pred = self.head(self.dropout(output.transpose(0, 1)))

        # добавить встраивание и позиционное кодирование к выходу
        pred['embedding'] = x_d
        pred['positional_encoding'] = positional_encoding

        return pred
