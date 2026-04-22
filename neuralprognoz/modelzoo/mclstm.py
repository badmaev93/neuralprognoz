from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.utils.config import Config


class MCLSTM(BaseModel):
    """Модель MC-LSTM, сохраняющая массу, из работы Hoedt et al. [1]_.

    Эта модель реализует точную конфигурацию MC-LSTM, которая использовалась Hoedt et al. [1]_ в гидрологическом эксперименте
    (для получения более общего и настраиваемого класса модели MC-LSTM см. официальный
    `репозиторий MC-LSTM на GitHub <https://github.com/ml-jku/mc-lstm>`_).

    MC-LSTM — это модель временных рядов, вдохновленная LSTM, которая гарантирует сохранение массы указанного `mass_input`
    благодаря особой конструкции своей архитектуры. Модель состоит из трех частей:

    - входной/junction gate, который распределяет массовый вход на определенном временном шаге по ячейкам памяти
    - матрица перераспределения, которая позволяет внутреннюю реорганизацию хранимой массы
    - выходной gate, который определяет долю хранимой массы, вычитаемую из ячеек памяти и определяет выход MC-LSTM

    Начиная с общей архитектуры MC-LSTM, представленной Hoedt et al. [1]_, наиболее заметной адаптацией для
    гидрологического применения является использование так называемой "*мусорной ячейки*". Мусорная ячейка — это
    одна конкретная ячейка вектора состояния ячейки, которая не используется для вывода прогноза модели,
    который определяется как сумма исходящей массы всех ячеек памяти, кроме мусорной ячейки. Для получения
    дополнительных сведений и различных вариантов, которые были протестированы для применения в гидрологии,
    см. Приложение B.4.2 в Hoedt et al [1]_.

    Аргумент конфигурации `head` игнорируется для этой модели, и прогноз модели всегда вычисляется как сумма
    исходящей массы (исключая выход мусорной ячейки).

    Аргумент конфигурации `initial_forget_bias` используется здесь для закрытия (отрицательные значения) или
    открытия (положительные значения) выходного gate в начале обучения. Закрытый выходной gate означает, что
    MC-LSTM должен активно учиться, когда удалять массу из системы, что можно рассматривать как аналог открытого
    forget gate в стандартном LSTM.
    LSTM.

    Чтобы использовать этот класс модели, вы должны указать имя массового входа с помощью аргумента конфигурации
    `mass_input`. Кроме того, массовый вход и целевая переменная *не* должны быть нормализованы. Используйте
    аргумент конфигурации `custom_normalization` и установите ключи `centering` и `scaling` для обоих в `None`
    (см. :doc:`аргументы конфигурации </usage/config>` для получения более подробной информации о
    `custom_normalization`).

    В настоящее время поддерживается только один массовый вход на временной шаг, а также одна цель.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Вызывает
    ------
    ValueError
        Если в конфигурации не указан или указан более одного `mass_input`. Кроме того, ошибка возникает, если скрытый
        size меньше 2, количество `target_variables` больше 1, или `dynamics_embedding` указан
        в конфигурации, что (в настоящее время) не поддерживается для этого класса модели.

    Ссылки
    ----------
    .. [1] Hoedt, P. J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., and
        Klambauer, G: MC-LSTM: Mass-Conserving LSTM, arXiv Preprint, https://arxiv.org/abs/2101.05186, 2021.
    """
    module_parts = ["embedding_net", "mclstm"]

    def __init__(self, cfg: Config):
        super(MCLSTM, self).__init__(cfg=cfg)

        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars > 1:
            raise ValueError("В настотящее время, MC-LSTM поддерживает только один массовый вход")
        elif self._n_mass_vars == 0:
            raise ValueError("В конфигурации не указан массовый вход. Укажите массовый вход с помощью аргумента конфигурации `mass_inputs`")

        if cfg.dynamics_embedding is not None:
            raise ValueError("В настоящее время, MC-LSTM не поддерживает встраивание для динамических входов")

        if cfg.hidden_size <= 1:
            raise ValueError("Для одной (обязательной) мусорной ячейки и одной массовой ячейки требуется скрытый размер не менее 2.")

        if len(cfg.target_variables) > 1:
            raise ValueError("В настоящее время, MC-LSTM поддерживает только одну цель.")

        self.embedding_net = InputLayer(cfg)

        n_aux_inputs = self.embedding_net.statics_output_size + self.embedding_net.dynamics_output_size
        self.mclstm = _MCLSTMCell(mass_input_size=self._n_mass_vars,
                                  aux_input_size=n_aux_inputs,
                                  hidden_size=cfg.hidden_size,
                                  cfg=cfg)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход через модель MC-LSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Выходные данные модели и промежуточные состояния в виде словаря.
                - `y_hat`: прогнозы модели формы [batch size, sequence length, 1].
                - `m_out`: массовый выход MC-LSTM (включая мусорную ячейку с индексом 0) формы [batch size, sequence
                  length, hidden size].
                - `c`: состояние ячейки MC-LSTM формы [batch size, sequence length, hidden size].

        """
        # возможно, пропустить статические входы через слои встраивания и объединить с динамическими.
        # Это не включает массовый вход.
        x_a = self.embedding_net(data, concatenate_output=True)

        # транспонировать в [seq_length, batch_size, n_features].
        x_m = torch.cat([data['x_d'][k] for k in self.cfg.mass_inputs], dim=-1).transpose(0, 1)

        # выполнить прямой проход через ячейку MC-LSTM
        m_out, c = self.mclstm(x_m, x_a)

        # исключить мусорную ячейку из прогнозов модели (см. связанную публикацию для получения подробной информации.)
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)

        return {'y_hat': output.transpose(0, 1), 'm_out': m_out.transpose(0, 1), 'c': c.transpose(0, 1)}


class _MCLSTMCell(nn.Module):
    """Логика ячейки MC-LSTM"""

    def __init__(self, mass_input_size: int, aux_input_size: int, hidden_size: int, cfg: Config):

        super(_MCLSTMCell, self).__init__()
        self.cfg = cfg
        self._hidden_size = hidden_size

        gate_inputs = aux_input_size + hidden_size + mass_input_size

        # инициализация гейтов
        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.input_gate = _NormalizedGate(in_features=gate_inputs,
                                          out_shape=(mass_input_size, hidden_size),
                                          normalizer="normalized_sigmoid")
        self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                              out_shape=(hidden_size, hidden_size),
                                              normalizer="normalized_relu")

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=self.cfg.initial_forget_bias)

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Выполняет прямой проход через ячейку MC-LSTM.

        Параметры
        ----------
        x_m : torch.Tensor
            Массовый вход, который будет сохранен сетью.
        x_a : torch.Tensor
            Вспомогательные входы, которые будут использоваться для модуляции гейтов, но информация о которых не будет храниться внутри ячеек MC-LSTM.

        Возвращает
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Массовый и памяти ячейки за каждый временной шаг формы [sequence length, batch size, hidden size]

        """
        _, batch_size, _ = x_m.size()
        ct = x_m.new_zeros((batch_size, self._hidden_size))

        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)

            m_out.append(mt_out)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)

        return m_out, c

    def _step(self, xt_m, xt_a, c):
        """ Сделать один временной шаг MCLSTM. """
        # во всех доступных данных используется для получения активации гейтов. Состояния ячеек
        # L1-нормализуются, чтобы растущие состояния ячеек в течение последовательности не вызывали проблем в гейтах.
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # рассчитать gate активации
        i = self.input_gate(features)
        r = self.redistribution(features)
        o = self.output_gate(features)

        # Распределить поступающую массу по состояниям ячеек
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # перетасовать массу в состояниях ячеек с использованием матрицы перераспределения
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)

        # вычислить новые массовые состояния
        m_new = m_in + m_sys

        # вернуть исходящую массу и вычесть это значение из состояний ячеек.
        return o * m_new, (1 - o) * m_new


class _Gate(nn.Module):
    """Вспомогательный класс для реализации стандартного сигмоидного гейта"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через нормализованный гейт"""
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """Вспомогательный класс для реализации гейта с нормализованной функцией активации"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Неизвестный нормализатор {normalizer}. Должен быть одним из {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход через нормализованный гейт"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)
