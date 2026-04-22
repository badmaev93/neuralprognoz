from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config


class SequentialForecastLSTM(BaseModel):
    """Модель прогнозирования, использующая одну последовательность LSTM с несколькими слоями встраивания.

    Это модель прогнозирования, которая использует одну последовательную (LSTM) модель, 
    которая разворачивается как по последовательности предыстории, так и по последовательности прогноза. 
    Отличие от стандартного ``CudaLSTM`` заключается в том, что (1) эта модель использует входные признаки 
    как предыстории, так и прогноза, и (2) она использует отдельную сеть встраивания для периода предыстории
    и периода прогноза. 
    
    Не используйте эту модель с ``forecast_overlap`` > 0.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Ошибки
    ------
    ValueError если forecast_overlap > 0
    ValueError если forecast и hindcast embedding nets имеют разные выходные размеры.
    """
    # Укажите субмодули модели, которые могут быть использованы для дообучения. Имена должны совпадать с атрибутами класса
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(SequentialForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Прогноз перекрытия не может быть установлен для модели прогнозирования последовательности. '
                             'Пожалуйста, установите значение None или удалите из файла конфигурации.')

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError('Forecast и hindcast embedding сети должны иметь одинаковый выходной размер при использовании модели прогнозирования последовательности.')

        self.lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=cfg.hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход на модели SequentialForecastLSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий выходные данные модели и промежуточные состояния.
                - lstm_output_hindcast: Выходная последовательность из LSTM предыстории.
                - lstm_output_forecast: Выходная последовательность из LSTM прогноза.
                - h_n_hindcast: Конечное скрытое состояние модели предыстории.
                - c_n_hindcast: Конечное состояние ячейки модели предыстории.
                - h_n_forecast: Конечное скрытое состояние модели прогноза.
                - c_n_forecast: Конечное состояние ячейки модели прогноза.
                - y_hat: Прогнозы последовательности из головного слоя.
        """
        # возможно, передать динамические и статические входы через слои встраивания, затем объединить их
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # выполнить часть LSTM предыстории
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.lstm(input=x_h)
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)

        # выполнить часть LSTM прогноза
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.lstm(x_f, (h_n_hindcast, c_n_hindcast))
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)

        # выполнить головной слой
        concatenated_predictions = torch.cat([lstm_output_hindcast, lstm_output_forecast], dim=1)
        pred = self.head(self.dropout(concatenated_predictions))

        # изменить форму на [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)
        h_n_forecast = h_n_forecast.transpose(0, 1)
        c_n_forecast = c_n_forecast.transpose(0, 1)

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_forecast': lstm_output_forecast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,

                'h_n_forecast': h_n_forecast,
                'c_n_forecast': c_n_forecast,
            }
        )

        return pred
