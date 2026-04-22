from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from neuralprognoz.training.regularization import BaseRegularization
from neuralprognoz.utils.config import Config

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class BaseLoss(torch.nn.Module):
    """Базовый класс потерь.

    Все потери расширяют этот класс, реализуя `_get_loss`.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    prediction_keys : List[str]
        Список прогнозируемых ключей. Во время прямого прохода переданный `prediction` словарь
        должен содержать эти ключи. Перечисленные здесь ключи не должны содержать идентификатор частоты.
    ground_truth_keys : List[str]
        Список ключей истинности, необходимых для вычисления потерь. Во время прямого прохода
        передаваемый словарь `data` должен содержать эти ключи. перечисленные здесь ключи должны быть без
        частотного идентификатора.
    additional_data : List[str], optional
        Дополнительный список ключей, которые будут извлечены из `data` в прямом проходе для вычисления потерь.
        Например, этот параметр может использоваться для передачи дисперсий, необходимых для вычисления NSE.
    output_size_per_target : int, optional
        Количество выходов модели (для каждого элемента в `prediction_keys`), подключенных к одной целевой переменной, по умолчанию равно 1. 
        Например, для регрессии один выходной элемент (последняя размерность в `y_hat`) соответствует одной целевой переменной. Для моделей
        смесей (например, GMM и CMAL) количество выходов на одну целевую переменную соответствует количеству распределений (`n_distributions`).
    """

    def __init__(self,
                 cfg: Config,
                 prediction_keys: List[str],
                 ground_truth_keys: List[str],
                 additional_data: List[str] = None,
                 output_size_per_target: int = 1):
        super(BaseLoss, self).__init__()
        self._predict_last_n = _get_predict_last_n(cfg)
        self._frequencies = [f for f in self._predict_last_n.keys() if f not in cfg.no_loss_frequencies]
        self._output_size_per_target = output_size_per_target

        self._regularization_terms = []

        # Названия ключей истинных значений и предсказаний, которые должны быть распакованы и отобраны для последних n элементов.
        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        # Подклассы могут использовать этот список для регистрации входных данных, которые будут распаковываться во время прямого прохода
        # и передаваться в качестве именованных аргументов (kwargs) методу _get_loss() без предварительного отбора подмножества.
        self._additional_data = []
        if additional_data is not None:
            self._additional_data = additional_data

        # Все классы позволяют устанавливать весовые коэффициенты для каждой цели в настройках с несколькими целями. 
        # По умолчанию все цели имеют равный вес.
        if cfg.target_loss_weights is None:
            weights = torch.tensor([1 / len(cfg.target_variables) for _ in range(len(cfg.target_variables))])
        else:
            if len(cfg.target_loss_weights) == len(cfg.target_variables):
                weights = torch.tensor(cfg.target_loss_weights)
            else:
                raise ValueError("Количество весовых коэффициентов должно быть равно количеству целевых переменных.")
        self._target_weights = weights

    def forward(self, prediction: Dict[str, torch.Tensor],
                data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Вычислить потери.

        Параметры
        ----------
        prediction : Dict[str, torch.Tensor]
            Словарь прогнозов для каждой частоты. Если прогнозируется больше одной частоты,
             ключи должны иметь суффиксы ``_{frequency}``. Требуемые ключи см. в документации конкретной потери.
        data : Dict[str, torch.Tensor]
            Словарь истинных данных для каждой частоты. Если предсказывается больше одной частоты,
            ключи должны иметь суффиксы ``_{frequency}``. Требуемые ключи см. в документации конкретной потери.

        Возвращает
        -------
        torch.Tensor
            Общая рассчитанная потеря.
        Dict[str, torch.Tensor]
            Индивидуальные компоненты потери (например, регуляризационные члены). 'total_loss' содержит общую потерю.
        """
        # Распаковка специфичных для потери дополнительных аргументов
        kwargs = {key: data[key] for key in self._additional_data}

        losses = []
        prediction_sub, ground_truth_sub = {}, {}
        for freq in self._frequencies:
            if self._predict_last_n[freq] == 0:
                continue  # нет прогнозов для данной частоты
            freq_suffix = '' if freq == '' else f'_{freq}'

            # Применять `predict_last_n` и маску ко всем выходам данной частоты одновременно.
            freq_pred, freq_gt = self._subset_in_time(
                {key: prediction[f'{key}{freq_suffix}'] for key in self._prediction_keys},
                {key: data[f'{key}{freq_suffix}'] for key in self._ground_truth_keys}, self._predict_last_n[freq])

            # Запоминать подмножества для компоненты с мультичастотами.
            prediction_sub.update({f'{key}{freq_suffix}': freq_pred[key] for key in freq_pred.keys()})
            ground_truth_sub.update({f'{key}{freq_suffix}': freq_gt[key] for key in freq_gt.keys()})

            for n_target, weight in enumerate(self._target_weights):
                # Выбрать подмножества выходных данных модели и истинных меток, соответствующих именно этой конкретной цели.
                target_pred, target_gt = self._subset_target(freq_pred, freq_gt, n_target)

                # Хук модели для выбора дополнительного набора данных, который может отличаться для различных функций потерь.
                kwargs_sub = self._subset_additional_data(kwargs, n_target)

                loss = self._get_loss(target_pred, target_gt, **kwargs_sub)
                losses.append(loss * weight)

        loss = torch.sum(torch.stack(losses))
        total_loss = loss.clone()
        all_losses = defaultdict(lambda: 0)
        all_losses['loss'] = loss
        for reg_module in self._regularization_terms:
            reg_out = reg_module(prediction_sub, ground_truth_sub,
                                 {k: v for k, v in prediction.items() if k not in self._prediction_keys})
            total_loss += reg_module.weight * reg_out
            # Одно и то же название может встречаться несколько раз. Суммируется все регуляризации с одинаковым названием для целей логирования.
            all_losses[reg_module.name] += reg_out
        all_losses['total_loss'] = total_loss
        return total_loss, all_losses

    @staticmethod
    def _subset_in_time(prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                        predict_last_n: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ground_truth_sub = {key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()}
        prediction_sub = {key: pred[:, -predict_last_n:, :] for key, pred in prediction.items()}

        return prediction_sub, ground_truth_sub

    def _subset_target(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                       n_target: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Определить, какие нейроны выхода соответствуют n-й целевой переменной
        start = n_target * self._output_size_per_target
        end = (n_target + 1) * self._output_size_per_target
        prediction_sub = {key: pred[:, :, start:end] for key, pred in prediction.items()}

        # Выделить цель путем срезания массива, чтобы сохранить форму `[batch_size, sequence_length, 1]`.
        ground_truth_sub = {key: gt[:, :, n_target:n_target + 1] for key, gt in ground_truth.items()}

        return prediction_sub, ground_truth_sub

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        # по умолчанию, ничего не произошло
        return additional_data

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def set_regularization_terms(self, regularization_modules: List[BaseRegularization]):
        """Регистрация переданных членов регуляризации, которые будут добавлены к функции потерь.

        Параметры
        ----------
        regularization_modules : List[BaseRegularization]
            Список функций регуляризации, которые будут добавлены к потере во время прямого прохода `forward`.
        """
        self._regularization_terms = regularization_modules


class MaskedMSELoss(BaseLoss):
    """Среднеквадратичная ошибка (MSE) потеря.

    Чтобы использовать эту потерю в прямом проходе, переданный словарь `prediction`
    должен содержать ключ ``y_hat``, а словарь `data` должен содержать ``y``.

    Параметры
    ----------
    cfg : Config
         Конфигурация запуска.
    """

    def __init__(self, cfg: Config):
        super(MaskedMSELoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = 0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2)
        return loss


class MaskedRMSELoss(BaseLoss):
    """"Корень среднеквадратичной ошибки (RMSE) потеря.

    Чтобы использовать эту потерю в прямом проходе, переданный словарь `prediction`
    должен содержать ключ ``y_hat``, а словарь `data` должен содержать ``y``.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """

    def __init__(self, cfg: Config):
        super(MaskedRMSELoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = torch.sqrt(0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2))
        return loss


class MaskedNSELoss(BaseLoss):
    """Потеря коэффициента эффективности модели Нэш-Сатклиффа, усредненного по бассейну.

    Чтобы использовать эту потерю в прямом проходе, переданный словарь `prediction` должен
    содержать ключ ``y_hat``, а словарь `data` должен содержать ``y`` и ``per_basin_target_stds``.

    Описание функции потерь доступно в [#]_ .

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    eps: float, optional
        Маленькая константа для численной устойчивости.

    Ссылки
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
       universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
       *Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019
    """

    def __init__(self, cfg: Config, eps: float = 0.1):
        super(MaskedNSELoss, self).__init__(cfg,
                                            prediction_keys=['y_hat'],
                                            ground_truth_keys=['y'],
                                            additional_data=['per_basin_target_stds'])
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]
        per_basin_target_stds = kwargs['per_basin_target_stds']
        # Расширить первую размерность до размера predict_last_n.
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction['y_hat'])[mask]

        squared_error = (y_hat - y)**2
        weights = 1 / (per_basin_target_stds + self.eps)**2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        # Здесь необходимо выбрать подмножество стандартного отклонения для каждого бассейна (per_basin_target_stds). 
        # Применяем срез, чтобы сохранить форму [batch_size, sequence_length, 1].
        return {key: value[:, :, n_target:n_target + 1] for key, value in additional_data.items()}


class MaskedGMMLoss(BaseLoss):
    """Среднее отрицательное логарифмическое правдоподобие для смеси гауссовских распределений (GMM).

    Эта потеря обеспечивает отрицательное логарифмическое правдоподобие для GMM, которое является стандартной функцией потерь для таких моделей.
    Данная реализация адаптирована из [#]_ .

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    eps : float, optional
        Маленькая константа для численной устойчивости.

    Ссылки
    ----------
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    def __init__(self, cfg: Config, eps: float = 1e-10):
        super(MaskedGMMLoss, self).__init__(cfg,
                                            prediction_keys=['mu', 'sigma', 'pi'],
                                            ground_truth_keys=['y'],
                                            output_size_per_target=cfg.n_distributions)
        self.eps = eps

    @staticmethod
    def _gaussian_distribution(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Создать |mu|=K копий вектора y, вычесть среднее значение mu, разделить на стандартное отклонение sigma
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONE_OVER_2PI_SQUARED

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        s = prediction['sigma'][mask]
        p = prediction['pi'][mask]

        result = self._gaussian_distribution(m, s, y) * p
        result = torch.sum(result, dim=-1)
        result = -torch.log(result + self.eps)  # устойчивость к численным ошибкам ("epsilon-стабильность")
        return torch.mean(result)


class MaskedCMALLoss(BaseLoss):
    """Среднее отрицательное логарифмическое правдоподобие для модели с головным модулем CMAL. 
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    eps : float, optional
        Маленькая константа для численной устойчивости.
    """

    def __init__(self, cfg: Config, eps: float = 1e-8):
        super(MaskedCMALLoss, self).__init__(cfg,
                                             prediction_keys=['mu', 'b', 'tau', 'pi'],
                                             ground_truth_keys=['y'],
                                             output_size_per_target=cfg.n_distributions)
        self.eps = eps  # epsilon-стабильность

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]
        t = prediction['tau'][mask]
        p = prediction['pi'][mask]

        error = y - m
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b
        log_weights = torch.log(p + self.eps)

        result = torch.logsumexp(log_weights + log_like, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class MaskedUMALLoss(BaseLoss):
    """Среднее отрицательное логарифмическое правдоподобие для модели с головным модулем UMAL.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    eps : float, optional
         Маленькая константа для численной устойчивости.
    """

    def __init__(self, cfg, eps: float = 1e-5):
        super(MaskedUMALLoss, self).__init__(cfg,
                                             prediction_keys=['mu', 'b'],
                                             ground_truth_keys=['y_extended', 'tau'],
                                             output_size_per_target=2)
        self.eps = eps
        self._n_taus_count = cfg.n_taus
        self._n_taus_log = torch.as_tensor(np.log(cfg.n_taus).astype('float32'))

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y_extended']).any(1).any(1)
        y = ground_truth['y_extended'][mask]
        t = ground_truth['tau'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]

        # расчет логарифмического правдоподобия
        error = y - m
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b

        original_batch_size = int(log_like.shape[0] / self._n_taus_count)
        log_like_split = torch.cat(log_like[:, :, :].split(original_batch_size, 0), 2)

        result = torch.logsumexp(log_like_split, dim=2) - self._n_taus_log
        result = -torch.mean(torch.sum(result, dim=1))
        return result


def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {'': list(predict_last_n.values())[0]}  # Если присутствует только одна частота, опускаем её идентификатор.






    return predict_last_n

class MaskedSSigmaDeltaLoss(BaseLoss):
    """
    Прямая оптимизация отношения S/sigma_delta.
    
    Цель: Минимизировать отношение RMSE модели к RMSE инерционного прогноза.
    Если Loss < 1.0, значит модель работает лучше, чем "наивный" прогноз.
    """

    def __init__(self, cfg: Config, eps: float = 1e-6):
        super(MaskedSSigmaDeltaLoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        # 1. Получаем маску валидных данных
        y = ground_truth['y']
        y_hat = prediction['y_hat']
        mask = ~torch.isnan(y)
        
        # 2. Считаем S (RMSE модели)
        # S = sqrt(mean((y - y_hat)^2))
        mse_model = torch.mean((y_hat[mask] - y[mask])**2)
        s_metric = torch.sqrt(mse_model + self.eps)

        # 3. Считаем Sigma Delta (RMSE инерционного прогноза на этом же батче)
        # Нам нужно посчитать разницу между y[t] и y[t-1] внутри последовательности
        # y имеет форму [batch, seq_len, 1]
        
        # Берем срез с 1-го элемента по текущий и с 0-го по предпоследний
        # Это аналог np.diff по оси времени (dim=1)
        y_curr = y[:, 1:, :]
        y_prev = y[:, :-1, :]
        
        # Маска тоже уменьшается на 1
        mask_diff = mask[:, 1:, :] & mask[:, :-1, :]
        
        if torch.sum(mask_diff) == 0:
            # Если батч слишком маленький или одни дыры, возвращаем просто MSE, чтобы не упасть
            return mse_model

        # Инерционная ошибка: (y[t] - y[t-1])^2
        # Это и есть квадрат Sigma_delta (если считать мат. ожидание изменений 0)
        diff_squared = (y_curr[mask_diff] - y_prev[mask_diff])**2
        sigma_delta_sq = torch.mean(diff_squared)
        sigma_delta = torch.sqrt(sigma_delta_sq + self.eps)

        # 4. Итоговый лосс: Отношение
        loss = s_metric / sigma_delta
        
        return loss


class MaskedHybridDiffLoss(BaseLoss):
    """
    Гибридный лосс: MSE + Штраф за форму (производную).

    Помогает S/sigma тем, что заставляет модель предсказывать не только значение,
    но и *направление изменения* (рост или спад). Это убирает лаги, которые портят S/sigma.

    Loss = MSE + alpha * Diff_MSE
    """

    def __init__(self, cfg: Config, alpha: float = 1.0):
        super(MaskedHybridDiffLoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])
        self.alpha = alpha # Вес штрафа за производную

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        y = ground_truth['y']
        y_hat = prediction['y_hat']
        mask = ~torch.isnan(y)

        # 1. Классический MSE (базовая точность)
        loss_mse = torch.mean((y_hat[mask] - y[mask])**2)

        # 2. Diff Loss (Ошибка производной)
        # Заставляем модель правильно угадывать: "завтра вода поднимется или опустится?"
        # Diff_Fact = y[t] - y[t-1]
        # Diff_Pred = y_hat[t] - y_hat[t-1]

        y_diff_true = y[:, 1:, :] - y[:, :-1, :]
        y_diff_pred = y_hat[:, 1:, :] - y_hat[:, :-1, :]

        mask_diff = mask[:, 1:, :] & mask[:, :-1, :]

        if torch.sum(mask_diff) > 0:
            loss_diff = torch.mean((y_diff_pred[mask_diff] - y_diff_true[mask_diff])**2)
        else:
            loss_diff = 0.0

        # Сумма
        return loss_mse + self.alpha * loss_diff


class MaskedPhysicsInformedLoss(BaseLoss):
    """Физически-информированная функция потерь для RCPIKLA (Zhou & Liu 2025, Eq. 18-19).

    L_combined = α · MSE(Q̂, Q_obs) + β · L_phys
    L_phys     = mean( max(Mt − Q̂_i, 0) )  — штраф когда Q̂ < снеготаяние

    Физический смысл: расход не может быть меньше объёма таяния снега.
    snow_melt передаётся моделью как 'snow_melt' в словаре prediction.
    Если snow_melt отсутствует (нет SnowpackLayer), работает как обычный MSE.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска. Считывает phys_loss_alpha и phys_loss_beta.
    """

    def __init__(self, cfg: Config):
        super(MaskedPhysicsInformedLoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
        )
        self.alpha = cfg.phys_loss_alpha
        self.beta  = cfg.phys_loss_beta

    def forward(self, prediction: Dict[str, torch.Tensor],
                data: Dict[str, torch.Tensor]):
        # Сохраняем snow_melt из prediction перед вызовом родительского forward,
        # т.к. BaseLoss._subset_in_time работает только с prediction_keys=['y_hat'].
        self._snow_melt_raw = prediction.get('snow_melt', None)
        return super().forward(prediction, data)

    def _get_loss(self, prediction: Dict[str, torch.Tensor],
                  ground_truth: Dict[str, torch.Tensor], **kwargs):
        y     = ground_truth['y']
        y_hat = prediction['y_hat']
        mask  = ~torch.isnan(y)

        # α · MSE
        mse = torch.mean((y_hat[mask] - y[mask]) ** 2)
        loss = self.alpha * mse

        # β · L_phys (только если есть snow_melt от SnowpackLayer)
        if self._snow_melt_raw is not None and self.beta > 0:
            # snow_melt: [batch, seq_len, 1]  (rcpikla.py добавляет unsqueeze(-1))
            melt = self._snow_melt_raw
            # Обрезаем до predict_last_n по оси времени (как BaseLoss делает с y_hat)
            T_hat = y_hat.shape[1]
            if melt.shape[1] > T_hat:
                melt = melt[:, -T_hat:, :]
            # L_phys = mean( relu(Mt - Q̂) )  — штраф за Q̂ < Mt
            l_phys = torch.mean(torch.relu(melt - y_hat))
            loss = loss + self.beta * l_phys

        return loss


class SummerWeightedPhysicsLoss(BaseLoss):
    """Физически-информированный loss с повышенным весом для летних месяцев (июль–сентябрь).

    Мотивация: S/σ_delta считается только для Q3 (июль–сентябрь),
    поэтому имеет смысл сильнее штрафовать ошибки именно в эти месяцы.

    L = α · MSE_base + γ · MSE_summer + β · L_phys
    где MSE_summer — MSE только по шагам с month ∈ {7, 8, 9}.

    Конфигурационные ключи:
        phys_loss_alpha  : вес базового MSE (default 1.0)
        phys_loss_beta   : вес физического штрафа (default 0.5)
        summer_loss_gamma: доп. вес летних месяцев (default 2.0)
    """

    def __init__(self, cfg: Config):
        super(SummerWeightedPhysicsLoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
        )
        self.alpha = cfg.phys_loss_alpha
        self.beta  = cfg.phys_loss_beta
        self.gamma = cfg.summer_loss_gamma
        self._summer_months = {7, 8, 9}

    def forward(self, prediction: Dict[str, torch.Tensor],
                data: Dict[str, torch.Tensor]):
        self._snow_melt_raw = prediction.get('snow_melt', None)
        self._dates_raw     = data.get('date', None)
        return super().forward(prediction, data)

    def _get_loss(self, prediction: Dict[str, torch.Tensor],
                  ground_truth: Dict[str, torch.Tensor], **kwargs):
        import numpy as np
        import pandas as pd

        y     = ground_truth['y']     # [B, predict_last_n, 1]
        y_hat = prediction['y_hat']
        mask  = ~torch.isnan(y)

        mse_base = torch.mean((y_hat[mask] - y[mask]) ** 2)
        loss = self.alpha * mse_base

        # доп. штраф за летние месяцы (июль–сентябрь)
        if self._dates_raw is not None and self.gamma > 0:
            try:
                dates_np = self._dates_raw
                if hasattr(dates_np, 'numpy'):
                    dates_np = dates_np.numpy()
                T = y.shape[1]
                if dates_np.ndim == 2 and dates_np.shape[1] >= T:
                    dates_np = dates_np[:, -T:]
                months = np.vectorize(lambda d: pd.to_datetime(d).month)(dates_np)
                summer_mask_np = np.isin(months, list(self._summer_months))
                summer_mask = torch.from_numpy(summer_mask_np).unsqueeze(-1).to(y.device)
                combined = mask & summer_mask
                if combined.any():
                    mse_summer = torch.mean((y_hat[combined] - y[combined]) ** 2)
                    loss = loss + self.gamma * mse_summer
            except Exception:
                pass

        # физический штраф L_phys
        if self._snow_melt_raw is not None and self.beta > 0:
            melt = self._snow_melt_raw
            T_hat = y_hat.shape[1]
            if melt.shape[1] > T_hat:
                melt = melt[:, -T_hat:, :]
            l_phys = torch.mean(torch.relu(melt - y_hat))
            loss = loss + self.beta * l_phys

        return loss


class JulyWeightedPhysicsLoss(BaseLoss):
    """Физически-информированный loss с повышенным весом для первого шага прогноза (июль).

    predict_last_n=3: шаги [0,1,2] = [июль, август, сентябрь].
    Первый шаг (июль) получает вес july_loss_weight, остальные — 1.0.

    L = alpha * (w_july*MSE_step0 + MSE_step1 + MSE_step2) / (w_july+2)
        + gamma * MSE_summer_base + beta * L_phys

    Конфигурационные ключи:
        phys_loss_alpha   : общий вес MSE (default 1.0)
        phys_loss_beta    : вес физического штрафа (default 0.5)
        summer_loss_gamma : доп. вес летних месяцев (default 1.5)
        july_loss_weight  : вес первого шага (июль) (default 4.0)
    """

    def __init__(self, cfg: Config):
        super(JulyWeightedPhysicsLoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
        )
        self.alpha  = cfg.phys_loss_alpha
        self.beta   = cfg.phys_loss_beta
        self.gamma  = cfg.summer_loss_gamma
        self.w_july = cfg.july_loss_weight
        self._summer_months = {7, 8, 9}

    def forward(self, prediction, data):
        self._snow_melt_raw = prediction.get('snow_melt', None)
        self._dates_raw     = data.get('date', None)
        return super().forward(prediction, data)

    def _get_loss(self, prediction, ground_truth, **kwargs):
        import numpy as np
        import pandas as pd

        y     = ground_truth['y']     # [B, predict_last_n, 1]
        y_hat = prediction['y_hat']   # [B, predict_last_n, 1]
        mask  = ~torch.isnan(y)

        n_steps = y.shape[1]

        # Веса по шагам: w_july для шага 0 (июль), 1.0 для остальных
        step_weights = torch.ones(n_steps, device=y.device)
        step_weights[0] = self.w_july

        loss = torch.tensor(0.0, device=y.device)
        total_w = 0.0
        for s in range(n_steps):
            m_s = mask[:, s, :]
            if m_s.sum() == 0:
                continue
            mse_s = torch.mean((y_hat[:, s, :][m_s] - y[:, s, :][m_s]) ** 2)
            loss = loss + step_weights[s] * mse_s
            total_w += float(step_weights[s])
        if total_w > 0:
            loss = self.alpha * loss / total_w

        # Доп. летний штраф по датам (поверх базового MSE)
        if self._dates_raw is not None and self.gamma > 0:
            try:
                dates_np = self._dates_raw
                if hasattr(dates_np, 'numpy'):
                    dates_np = dates_np.numpy()
                T = y.shape[1]
                if dates_np.ndim == 2 and dates_np.shape[1] >= T:
                    dates_np = dates_np[:, -T:]
                months = np.vectorize(lambda d: pd.to_datetime(d).month)(dates_np)
                summer_mask_np = np.isin(months, list(self._summer_months))
                summer_mask = torch.from_numpy(summer_mask_np).unsqueeze(-1).to(y.device)
                combined = mask & summer_mask
                if combined.any():
                    mse_summer = torch.mean((y_hat[combined] - y[combined]) ** 2)
                    loss = loss + self.gamma * mse_summer
            except Exception:
                pass

        # Физический штраф
        if self._snow_melt_raw is not None and self.beta > 0:
            melt = self._snow_melt_raw
            T_hat = y_hat.shape[1]
            if melt.shape[1] > T_hat:
                melt = melt[:, -T_hat:, :]
            l_phys = torch.mean(torch.relu(melt - y_hat))
            loss = loss + self.beta * l_phys

        return loss