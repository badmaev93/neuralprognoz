from typing import Dict
import torch
import torch.nn as nn
from neuralprognoz.utils.config import Config
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.shm import SHM


class HybridModel(BaseModel):
    """Оболочка для объединения модели глубокого обучения с концептуальными гидрологическими моделями
    
    В текущей реализации моделью глубокого обучения всегда является LSTM. Концептуальная модель настраивается с помощью аргумента config
    `conceptual_model`. В настоящее время поддерживается `['SHM']`.

    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """

    def __init__(self, cfg: Config):
        super(HybridModel, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.conceptual_model = HybridModel._get_conceptual_model(cfg=cfg)

        self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=len(self.conceptual_model.parameter_ranges))

        self._reset_parameters()

    def _reset_parameters(self):
        """Специальная инициализация определенных весов модели."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход по модели.

        Параметры
        ----------
        data : Dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные объекты в виде пар "ключ-значение".

        Возвращает.
        -------
        Dict[str, torch.Tensor]
            Выходные данные модели, динамические параметры и промежуточные состояния, полученные из концептуальной модели
        """
       
        # запуск lstm
        x_d = self.embedding_net(data)
        lstm_output, _ = self.lstm(input=x_d)
        lstm_output = lstm_output.transpose(0, 1)  # переформировать на [batch_size, seq, n_hiddens]

        # сопоставить выходные данные lstm с размерностью параметров концептуальной модели
        lstm_out = lstm_output[:, self.cfg.warmup_period:, :]
        lstm_out = self.linear(lstm_out)

        # получить прогнозы
        # TODO: сделать так, чтобы концептуальная модель принимала набор входных данных, а не конкатенированный тензор.
        x_conceptual = torch.cat([data['x_d'][k][:, self.cfg.warmup_period:, :]
                                  for k in self.cfg.dynamic_conceptual_inputs], axis=-1)
        pred = self.conceptual_model(x_conceptual=x_conceptual, lstm_out=lstm_out)

        return pred

    @staticmethod
    def _get_conceptual_model(cfg: Config) -> BaseConceptualModel:
        """Получить концептуальную модель в зависимости от конфигурации запуска.

        Параметры
        ----------
        cfg : Config
            Конфигурация запуска.

        Возвращает
        -------
        BaseConceptualModel
            Новый экземпляр концептуальной модели типа, указанного в конфигурации.
        """
        if cfg.conceptual_model.lower() == "shm":
            conceptual_model = SHM(cfg=cfg)
        else:
            raise NotImplementedError(f"{cfg.conceptual_model} не реализована или не связана с `_get_conceptual_model()`")

        return conceptual_model
