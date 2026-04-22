from typing import List

import numpy as np
import torch
import torch.nn as nn


class FC(nn.Module):
    """Вспомогательный класс для построения (многослойных) fully-connected сетей.

    Этот класс используется для построения fully-connected embedding сетей для статических и/или динамических входных данных.
    Используйте аргумент конфигурации `statics/dynamics_embedding`, чтобы указать архитектуру embedding сети. См. класс
    `InputLayer`, чтобы узнать, как указать точную архитектуру embedding.

    Параметры
    ----------
    input_size : int
        Количество входных признаков.
    hidden_sizes : List[int]
        Размер скрытых и выходных слоев.
    activation : str, optional
        Функция активации для промежуточных слоев, по умолчанию tanh.
    dropout : float, optional
        Dropout rate в промежуточных слоях.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], activation: str = 'tanh', dropout: float = 0.0):
        super(FC, self).__init__()

        if len(hidden_sizes) == 0:
            raise ValueError('hidden_sizes должен содержать хотя бы один элемент для создания fully-connected сети.')

        self.output_size = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]

        activation = self._get_activation(activation)

        # создать сеть
        layers = []
        if hidden_sizes:
            for i, hidden_size in enumerate(hidden_sizes):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))

                layers.append(activation)
                layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(hidden_size, self.output_size))
        else:
            layers.append(nn.Linear(input_size, self.output_size))

        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _get_activation(self, name: str) -> nn.Module:
        if name.lower() == "tanh":
            activation = nn.Tanh()
        elif name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif name.lower() == "relu":
            activation = nn.ReLU()
        elif name.lower() == "linear":
            activation = nn.Identity()
        else:
            raise NotImplementedError(f"{name} в настоящее время не поддерживается в качестве функции активации в этом классе")
        return activation

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        for layer in self.net:
            if isinstance(layer, nn.modules.linear.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполнить прямой проход модели FC.

        Параметры
        ----------
        x : torch.Tensor
            Входные данные формы [any, any, input size]

        Возвращает
        -------
        torch.Tensor
            Векторы признаков формы [any, any, output_size], где 'output_size' - это размер последнего слоя сети.
        """
        return self.net(x)
