from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config
from neuralprognoz.modelzoo.fc import FC


class StackedForecastLSTM(BaseModel):
    """Прогнозная модель, использующая стекированные LSTM для предыстории и прогноза.

    Это прогнозная модель, которая использует две стекированные последовательные (LSTM) модели для обработки 
    предыстории и прогноза.
        
    Общая длина последовательности ``seq_length`` config parameter должна быть равна общей
    hindcast + forecast time ranges. 
    
    Длина прогнозной последовательности ``forecast_seq_length`` config parameter должна быть равна
    перекрывающейся части моделей предыстории и прогноза плюс прогнозный
    период. 
    
    Параметр ``forecast_overlap`` config parameter должен быть установлен на правильное перекрытие
    между этими двумя последовательностями. 
    
    Например, если мы хотим использовать период разгона в 365 дней для создания 7-дневного
    прогноза, то ``seq_length`` должен быть установлен на 372 (=365+7), ``forecast_seq_length``
    должен быть установлен на 365, а ``forecast_overlap`` должен быть установлен на 358 (=365-7). 
    
    Выходы из LSTM предыстории объединяются с входными 
    последовательностями для LSTM прогноза и сдвигаются во времени на горизонт прогноза (7 дней
    в приведенном выше примере. Это вызывает задержку между последними данными предыстории и
    новым временным шагом прогноза, что означает, что прогнозы не получают информацию от
    самых последних входных данных предыстории. Чтобы решить эту проблему, установите
    ``bidirectional_stacked_forecast_lstm`` config parameter в True, чтобы LSTM предыстории работала в двунаправленном режиме, и поэтому все выходы из LSTM предыстории
    получают информацию от самых последних входных данных предыстории.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # определить субмодули модели, которые могут быть использованы для тонкой настройки. Имена должны совпадать с атрибутами класса
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'forecast_lstm', 'hindcast_lstm', 'head']

    def __init__(self, cfg: Config):
        super(StackedForecastLSTM, self).__init__(cfg=cfg)

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_hidden_size = cfg.hindcast_hidden_size
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=self.hindcast_hidden_size,
            bidirectional=cfg.bidirectional_stacked_forecast_lstm
        )
        self.forecast_hidden_size = cfg.forecast_hidden_size
        forecast_input_size = self.forecast_embedding_net.output_size + self.hindcast_hidden_size
        if cfg.bidirectional_stacked_forecast_lstm:
            forecast_input_size += self.hindcast_hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=forecast_input_size,
            hidden_size=self.forecast_hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self.forecast_hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.hindcast_hidden_size:2 * self.hindcast_hidden_size] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.forecast_hidden_size:2 * self.forecast_hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход на модели StackedForecastLSTM.

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
                - y_hat: Предсказанные значения за весь период прогноза.

        Вызывает
        ------
        ValueError если длины последовательностей предыстории и прогноза не равны.
        """

        # возможно, передать динамические и статические входы через слои встраивания, а затем объединить их
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        if x_h.shape[0] != x_f.shape[0]:
            raise ValueError('Длины последовательностей предыстории и прогноза не равны.')

        # выполнить часть LSTM предыстории
        lstm_output_hindcast, _ = self.hindcast_lstm(input=x_h)

        # LSTM здесь НЕ является batch-first, поэтому транспонирование необходимо ниже.
        # Выход LSTM в этой точке имеет размер (seq, batch, dims).
        forecast_inputs = torch.concat((x_f, lstm_output_hindcast), dim=-1)  # конкатенация по переменной размерности

        # выполнить часть LSTM прогноза
        lstm_output_forecast, _ = self.forecast_lstm(forecast_inputs)
        
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)

        # выполнить head
        pred = self.head(self.dropout(lstm_output_forecast))

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_forecast': lstm_output_forecast,
            }
        )

        return pred
