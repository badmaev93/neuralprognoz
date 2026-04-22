import math

import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Класс для создания вектора позиционного кодирования для входных данных временных рядов в модель без явного временного измерения.

    Этот класс реализует вектор встраивания типа sin/cos с указанной максимальной длиной. Адаптировано из примера PyTorch:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Параметры
    ----------
    embedding_dim : int
        Размерность входных данных модели, которая обычно является выходом слоя встраивания.
    dropout : float
        Коэффициент отсева [0, 1), применяемый к вектору встраивания.
    max_len : int, optional
        Максимальная длина позиционного кодирования. Это должно быть больше наибольшей длины последовательности в выборке.
    """

    def __init__(self, embedding_dim, position_type, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_len * 2) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if position_type.lower() == 'concatenate':
            self._concatenate = True
        elif position_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(f"Нераспознанный тип позиционного кодирования: {position_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход для позиционного кодирования. Объединяет или добавляет позиционное кодирование к входным данным кодировщика.

        Параметры
        ----------
        x : torch.Tensor
            Размерность ``[sequence length, batch size, embedding output dimension]``.
            Данные, которые будут поданы на вход кодировщика после включения позиционного кодирования.
            Обычно это будет выход из слоя встраивания.

        Возвращает
        -------
        torch.Tensor
            Размерность ``[sequence length, batch size, encoder input dimension]``.
            Размерность входных данных кодировщика либо равна размерности выходных данных встраивания (если ``position_type == sum``),
            либо в два раза больше размерности выходных данных встраивания (если ``position_type == concatenate``).
            Входные данные кодировщика дополнены позиционным кодированием.

        """
        if self._concatenate:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
