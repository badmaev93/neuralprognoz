from typing import Dict

import pandas as pd
import torch

from neuralprognoz.datautils.utils import get_frequency_factor, sort_frequencies
from neuralprognoz.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Базовый класс для членов регуляризации

    Члены регуляризации подклассируют этот класс, реализуя метод `forward`.

    Parameters
    ----------
    cfg: Config
        Конфиуграция запуска.
    name: str
        Название регулярного члена.
    weight: float, optional.
        Вес регулярного члена. Значение по умолчанию: 1.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Рассчитать регуляризационное выражение.

        Возвращает
        ----------
        prediction : Dict[str, torch.Tensor]
            Словарь предикторов для каждой частоты. Если прогнозируется более одной частоты,
            ключи должны иметь суффикс ``_{frequency}``. . Для обязательных ключей - к документации
            конкретного метода потерь.
        ground_truth : Dict[str, torch.Tensor]
            Словарь эталонных значений для каждой частоты. Если прогнозируется более одной частоты,
            ключи должны иметь суффикс ``_{frequency}``.  Для обязательных ключей - к документации
            конкретного метода потерь.
        other_model_data : Dict[str, torch.Tensor]
            Словарь остальных пар ключ-значение в словаре прогнозов, которые непосредственно не связаны с
            прогнозами модели, но могут быть полезны для целей регуляризации, например внутренние сети, веса и т.п.
            
        Возвращает
        -------
        torch.Tensor
            Значение регуляризации.
        """
        raise NotImplementedError


class TiedFrequencyMSERegularization(BaseRegularization):
    """Регуляризация, штрафующая за несогласованные прогнозы по разным частотам.

    Данная регуляризация применима только тогда, когда прогнозируются хотя бы две частоты. Для каждой пары соседних
    частот f и f', где f имеет большую частоту, чем f', агрегированные прогнозы f приводятся к частоте f'
    и рассчитывается средне-квадратическое отклонение между f' и агрегированными значениями f.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    weight: float, optional.
        Вес регулярного члена. Значение по умолчанию: 1.

    Raises
    ------
    ValueError
        Если конфигурация запуска предусматривает прогнозирование только одной частоты.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(TiedFrequencyMSERegularization, self).__init__(cfg, name='tie_frequencies', weight=weight)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies])

        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization требует как минимум 2 частоты")

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                *args) -> torch.Tensor:
        """Вычисляет сумму средних квадратных отклонений между соседними прогнозируемыми частотами.

        Параметры
        ----------
        prediction : Dict[str, torch.Tensor]
            Словарь, содержащий ``y_hat_{frequency}`` для каждой частоты.
        ground_truth : Dict[str, torch.Tensor]
            Словарь, содержащий ``y_{frequency}`` для каждой частоты.

        Возвращает
        -------
        torch.Tensor
            Сумма средних квадратных отклонений для каждой пары соседних частот.
        """

        loss = 0
        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            frequency_factor = int(get_frequency_factor(self._frequencies[idx - 1], freq))
            freq_pred = prediction[f'y_hat_{freq}']
            mean_freq_pred = freq_pred.view(freq_pred.shape[0], freq_pred.shape[1] // frequency_factor,
                                            frequency_factor, -1).mean(dim=2)
            lower_freq_pred = prediction[f'y_hat_{self._frequencies[idx - 1]}'][:, -mean_freq_pred.shape[1]:]
            loss = loss + torch.mean((lower_freq_pred - mean_freq_pred)**2)

        return loss

class ForecastOverlapMSERegularization(BaseRegularization):
    """Регуляризация среднеквадратической ошибки для штрафа за разницу между ретропрогнозом и основным прогнозом.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(ForecastOverlapMSERegularization, self).__init__(cfg, name='forecast_overlap', weight=weight)

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_output: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Вычисляет квадратичную разницу между ретропрогнозом и основным прогнозом на перекрывающихся участках.

        Не работает с моделями с несколькими частотами.

        Параметры
        ----------
        prediction : Dict[str, torch.Tensor]
            Не используется.
        ground_truth : Dict[str, torch.Tensor]
            Не используется
        other_model_output : Dict[str, Dict[str, torch.Tensor]]
            Словарь, содержащий ``y_forecast_overlap`` и ``y_hindcast_overlap``,
            оба являются словарями, содержащими ключи к соответствующим выводам модели.

        Возвращает
        -------
        torch.Tensor
            Сумма среднеквадратических отклонений между пересекающимися участками ретропрогноза и основного прогноза.

        Raises
        ------
        ValueError если y_hindcast_overlap или y_forecast_overlap отсутствует в выводах модели.
        """
        loss = 0
        if 'y_hindcast_overlap' not in other_model_output or not other_model_output['y_hindcast_overlap']:
            raise ValueError('y_hindcast_overlap не представлен в выходе модели.')
        if 'y_forecast_overlap' not in other_model_output or not other_model_output['y_forecast_overlap']:
            raise ValueError('y_forecast_overlap не представлен в выходе модели.')
        for key in other_model_output['y_hindcast_overlap']:
            hindcast = other_model_output['y_hindcast_overlap'][key]
            forecast = other_model_output['y_forecast_overlap'][key]
            loss += torch.mean((hindcast - forecast)**2)
        return loss


class SnowmeltConsistencyRegularization(BaseRegularization):
    """
    Регуляризация, штрафующая за сток, который меньше рассчитанного снеготаяния.
    L_phys = max(M_t - Q_i, 0)
    """
    def __init__(self, cfg: Config, weight: float = 1.0):
        super(SnowmeltConsistencyRegularization, self).__init__(cfg, name='snowmelt_consistency', weight=weight)

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        # Убедимся, что модель вернула 'snow_melt'
        if 'snow_melt' not in other_model_output:
            raise ValueError("Модель должна возвращать 'snow_melt' для этой регуляризации.")
            
        q_sim = prediction['y_hat']
        snow_melt = other_model_output['snow_melt']
        
        # Учитываем predict_last_n
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, dict): # Для мульти-частотного случая
             # Упрощение: используем первую частоту. Для полной поддержки нужен более сложный код.
            predict_last_n = list(predict_last_n.values())[0]

        q_sim = q_sim[:, -predict_last_n:, :]
        snow_melt = snow_melt[:, -predict_last_n:, :]

        # Штраф за физически некорректные предсказания
        # Усреднение по батчу и временной последовательности
        loss = torch.mean(torch.clamp(snow_melt - q_sim, min=0))
        
        return loss