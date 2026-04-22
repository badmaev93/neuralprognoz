
from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config
from neuralprognoz.modelzoo.fc import FC


class MultiHeadForecastLSTM(BaseModel):
    """Модель прогнозирования, которая не развертывается на горизонте прогнозирования.
    
    Это модель прогнозирования, которая запускает последовательную (LSTM) модель до времени выпуска прогноза, 
    а затем напрямую предсказывает последовательность временных шагов прогноза без использования рекуррентного развертывания.
    Прогнозирование осуществляется с помощью пользовательского слоя ``FC`` (полносвязного), который может включать глубину.

    Не используйте эту модель с ``forecast_overlap`` > 0.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    
    Вызывает
    ------
    ValueError если forecast_overlap > 0.
    ValueError если не указана forecast_network.
    """
    # укажите подмодули модели, которые позже можно будет использовать для дообучения (finetuning). Имена должны совпадать с атрибутами класса
    module_parts = [
        'forecast_mebedding_net',
        'hindcast_embedding_net',
        'hindcast_lstm',
        'forecast_network',
        'hindcast_head',
        'forecast_head'
    ]

    def __init__(self, cfg: Config):
        super(MultiHeadForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Перекрытие прогноза не может быть установлено для многоголовочной модели прогнозирования. '
                             'Пожалуйста, установите значение None или удалите из файла конфигурации.')

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=cfg.hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        if not cfg.forecast_network:
            raise ValueError('Многоголовочная модель прогнозирования требует указания сети прогнозирования в файле конфигурации.')

        input_size = self.forecast_embedding_net.output_size*cfg.forecast_seq_length + cfg.hidden_size
        forecast_network_output_size = cfg.forecast_network['hiddens'][-1] * cfg.forecast_seq_length
        self.forecast_network = FC(
            input_size=input_size,
            hidden_sizes=cfg.forecast_network['hiddens'][:-1] + [forecast_network_output_size],
            activation=cfg.forecast_network['activation'],
            dropout=cfg.forecast_network['dropout']
        )

        self.hindcast_head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
        self.forecast_head = get_head(cfg=cfg, n_in=cfg.forecast_network['hiddens'][-1], n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Особая инициализация некоторых весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход на модели MultiheadForecastLSTM.
        
        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.
        
        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий выходные данные модели и промежуточные состояния.
                - lstm_output_hindcast: Выходная последовательность из LSTM.
                - output_forecast: Прогнозы (перед выходным слоем) из периода прогнозирования.
                - h_n_hindcast: Конечное скрытое состояние модели.
                - c_n_hindcast: Конечное состояние ячейки модели.
                - y_hat: Прогнозы по последовательности из выходного слоя.
        """
        # возможно, передать динамические и статические входные данные через слои встраивания, затем объединить их
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # прогнать рекуррентную часть lstm
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(input=x_h)
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        output_hindcast = self.hindcast_head(self.dropout(lstm_output_hindcast))

        # изменить форму на [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)

        # прогнать головные слои прогнозирования
        batch_size = x_f.shape[1]
        x_f = x_f.transpose(0, 1).contiguous()
        x = torch.cat([h_n_hindcast.squeeze(dim=1), x_f.view(batch_size, -1)], dim=-1)
        x = self.forecast_network(x)
        x = x.view(batch_size, self.cfg.forecast_seq_length, -1)
        output_forecast = self.forecast_head(self.dropout(x))

        # начать выходной словарь
        pred = {key: torch.cat([output_hindcast[key], output_forecast[key]], dim=1) for key in output_hindcast}

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'output_forecast': output_forecast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,
            }
        )

        return pred
