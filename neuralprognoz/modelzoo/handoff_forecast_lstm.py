from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.utils.config import Config
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.fc import FC
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.inputlayer import InputLayer

class HandoffForecastLSTM(BaseModel):
    """Модель LSTM с передачей состояния (state-handoff) для прогнозирования.

    Это модель прогнозирования, которая использует передачу состояния для перехода от модели последовательности 
    исторических данных (hindcast) к модели последовательности прогноза (forecast). Модель hindcast работает от 
    прошлого до настоящего (времени выпуска прогноза), а затем передает состояние ячейки и скрытое состояние 
    LSTM в (нелинейную) сеть передачи состояния (handoff network), которая затем используется для инициализации 
    состояния ячейки и скрытого состояния новой LSTM, которая разворачивается на период прогноза. 
    Сеть передачи состояния реализована как пользовательский слой FC, который может иметь несколько слоев. Сеть 
    передачи состояния реализована с использованием параметра конфигурации ``state_handoff_network``. LSTM для hindcast и 
    forecast имеют разные веса и смещения, разные головы и разные сети эмбеддингов. Размер скрытого состояния 
    LSTM для hindcast устанавливается с использованием параметра конфигурации ``hindcast_hidden_size``, а размер 
    скрытого состояния LSTM для forecast устанавливается с использованием параметра конфигурации ``forecast_hidden_size``.

    Модель HandoffForecastLSTM может реализовывать отложенную передачу состояния, при которой передача между LSTM 
    для hindcast и forecast происходит до времени выпуска прогноза. Это контролируется параметром ``forecast_overlap`` 
    в файле конфигурации, и LSTM для forecast и hindcast будут работать одновременно в течение количества временных шагов, 
    указанного в ``forecast_overlap``. Мы рекомендуем использовать опцию регуляризации ``ForecastOverlapMSERegularization`` 
    для регуляризации функции потерь путем (не)согласия между перекрывающимися частями LSTM для hindcast и forecast.
    Этот член регуляризации может быть запрошен путем установки параметра ``regularization`` в файле конфигурации, 
    включив
    ``forecast_overlap``.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    
    Вызывает
    ------
    ValueError если state_handoff_network не указан.
    """
    # указать субмодули модели, которые позже можно будет использовать для дообучения (finetuning). Имена должны совпадать с атрибутами класса
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'hindcast_lstm', 'forecast_lstm', 'hindcast_head', 'forecast_head', 'handoff_net']

    def __init__(self, cfg: Config):
        super(HandoffForecastLSTM, self).__init__(cfg=cfg)

        self.initial_hindcast_seq_length = self.cfg.seq_length - self.cfg.forecast_seq_length

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_hidden_size = cfg.hindcast_hidden_size
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=self.hindcast_hidden_size
        )
        self.forecast_hidden_size = cfg.forecast_hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=self.forecast_hidden_size
        )

        if not cfg.state_handoff_network:
            raise ValueError('Модель HandoffForecastLSTM требует указания сети передачи состояния в файле конфигурации.')

        self.handoff_net = FC(
            input_size=self.hindcast_hidden_size*2,
            hidden_sizes=cfg.state_handoff_network['hiddens'],
            activation=cfg.state_handoff_network['activation'],
            dropout=cfg.state_handoff_network['dropout']
        )
        self.handoff_linear =  FC(
            input_size=cfg.state_handoff_network['hiddens'][-1],
            hidden_sizes=[self.forecast_hidden_size*2],
            activation='linear',
            dropout=0.0
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.hindcast_head = get_head(cfg=cfg, n_in=self.hindcast_hidden_size, n_out=self.output_size)
        self.forecast_head = get_head(cfg=cfg, n_in=self.forecast_hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.hindcast_hidden_size:2 * self.hindcast_hidden_size] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.forecast_hidden_size:2 * self.forecast_hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход модели HandoffForecastLSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий выходные данные модели и промежуточные состояния.
                - lstm_output_hindcast: Выходная последовательность из LSTM для hindcast.
                - lstm_output_hindcast_overlap: Выходная последовательность из модели hindcast за период перекрытия между LSTM для forecast и hindcast.
                - lstm_output_forecast_overlap: Выходная последовательность из модели forecast за период перекрытия между LSTM для forecast и hindcast.
                - lstm_output_forecast: Выходная последовательность из LSTM для forecast.
                - y_forecast: Предсказания (после слоя head) за период прогноза.
                - y_forecast_overlap: Предсказания модели forecast за период перекрытия между LSTM для forecast и hindcast.
                - y_hindcast_overlap: Предсказания модели hindcast за период перекрытия между LSTM для forecast и hindcast.
                - y_hindcast: Предсказания за период hindcast.
                - h_n_hindcast: Финальное скрытое состояние модели hindcast.
                - c_n_hindcast: Финальное скрытое состояние модели hindcast.
                - h_n_handoff: Начальное скрытое состояние модели forecast.
                - c_n_handoff: Начальное скрытое состояние модели forecast.
                - h_n_forecast: Финальное скрытое состояние модели forecast.
                - c_n_forecast: Финальное скрытое состояние модели forecast.
                - y_hat: Предсказания за период из слоя head. Это конкатенация hindcast и forecast, причем для периода перекрытия берется hindcast.
        """
        # возможно, пропустить динамические и статические входные данные через embedding слои, а затем объединить их
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # запустить LSTM для hindcast
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(x_h[:self.initial_hindcast_seq_length, ...])
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        if x_h.shape[0] > self.initial_hindcast_seq_length:
            lstm_output_hindcast_overlap, _ = self.hindcast_lstm(x_h[self.initial_hindcast_seq_length:, ...], (h_n_hindcast, c_n_hindcast))
            lstm_output_hindcast_overlap = lstm_output_hindcast_overlap.transpose(0, 1)
        else:
            lstm_output_hindcast_overlap = None

        # передать начальное состояние в LSTM для forecast
        x = self.handoff_net(torch.cat([h_n_hindcast, c_n_hindcast], -1))
        initial_state = self.handoff_linear(x)
        h_n_handoff, c_n_handoff = initial_state.chunk(2, -1)
        h_n_handoff = h_n_handoff.contiguous()
        c_n_handoff = c_n_handoff.contiguous()

        # запустить LSTM для forecast
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.forecast_lstm(x_f, (h_n_handoff, c_n_handoff))
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)
        lstm_output_forecast_overlap = lstm_output_forecast[:, :self.cfg.forecast_overlap, :]
        lstm_output_forecast = lstm_output_forecast[:, self.cfg.forecast_overlap:, :]

        # запустить головы для hindcast и forecast
        y_hindcast = self.hindcast_head(self.dropout(lstm_output_hindcast))
        y_forecast = self.forecast_head(self.dropout(lstm_output_forecast))
        if x_h.shape[0] > self.initial_hindcast_seq_length:
            y_hindcast_overlap = self.hindcast_head(self.dropout(lstm_output_hindcast_overlap))
            y_forecast_overlap = self.forecast_head(self.dropout(lstm_output_forecast_overlap))
            pred = {key: torch.cat([y_hindcast[key], y_hindcast_overlap[key], y_forecast[key]], dim=1) for key in y_hindcast}
        else:
            pred = {key: torch.cat([y_hindcast[key], y_forecast[key]], dim=1) for key in y_hindcast}
            y_hindcast_overlap, y_forecast_overlap = None, None

        # изменить форму на [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)
        h_n_handoff = h_n_handoff.transpose(0, 1)
        c_n_handoff = c_n_handoff.transpose(0, 1)
        h_n_forecast = h_n_forecast.transpose(0, 1)
        c_n_forecast = c_n_forecast.transpose(0, 1)

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_hindcast_overlap': lstm_output_hindcast_overlap,
                'lstm_output_forecast_overlap': lstm_output_forecast_overlap,
                'lstm_output_forecast': lstm_output_forecast,

                'y_forecast': y_forecast,
                'y_forecast_overlap': y_forecast_overlap,
                'y_hindcast_overlap': y_hindcast_overlap,
                'y_hindcast': y_hindcast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,

                'h_n_handoff': h_n_handoff,
                'c_n_handoff': c_n_handoff,

                'h_n_forecast': h_n_forecast,
                'c_n_forecast': c_n_forecast,
            }
        )

        return pred
