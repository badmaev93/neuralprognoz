from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import xarray
from numba import njit
from torch.distributions import Categorical

from neuralprognoz.utils.config import Config


def sample_pointpredictions(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int,
                            scaler: Dict[str, Union[pd.Series, xarray.Dataset]]) -> Dict[str, torch.Tensor]:
    """Самобытные подходы к точечному отбору образцов для методов оценки неопределенности.
    
    Функция предоставляет разные методы точечного отбора образцов для различных подходов оценки неопределённости
    (например, Гауссовы смеси (Gaussian Mixture Models, GMM), счётные смеси асимметричных лапласовцев 
    (Countable Mixtures of Asymmetric Laplacians, CMAL), несчётные смеси асимметричных лапласовцев 
    (Uncountable Mixtures of Asymmetric Laplacians, UMAL) и Монте-Карло дропаут (Monte Carlo Dropout, MCD);
    обратить внимание: MCD можно комбинировать с остальными методами, установив `mc_dropout` в `True` в файле конфигурации.)

    
    Есть также варианты обработки отрицательных точечных выборок, которые возникают при взятии образца
    из оценок неопределенности. На данный момент эта функциональность поддерживает: 
    (а) 'clip': прямое обрезание значений на уровне нуля и 
    (б) 'truncate': повторный выбор значений, находящихся ниже нуля. 
    
    Параметры
    ----------
    model : BaseModel
       Нейронная модель Prognoz, из которой производится выборка.
    data : Dict[str, torch.Tensor]
        Словарь, содержащий входные признаки в виде пар ключ-значение.
    n_samples : int
        Количество точечных выборок прогнозов, которые необходимо создать.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Масштабатор (нормализатор) запускаемого процесса.

    Returns
    -------
    Dict[str, torch.Tensor]
        Словарь, содержащий выбранные выходные данные модели для последних predict_last_n 
        (параметр конфигурации) временных шагов каждой частоты.
    """

    if model.cfg.head.lower() == "gmm":
        samples = sample_gmm(model, data, n_samples, scaler)
    elif model.cfg.head.lower() == "cmal":
        samples = sample_cmal(model, data, n_samples, scaler)
    elif model.cfg.head.lower() == "umal":
        samples = sample_umal(model, data, n_samples, scaler)
    elif model.cfg.head.lower() == "regression":
        samples = sample_mcd(model, data, n_samples, scaler)  # regression head assumes mcd
    else:
        raise NotImplementedError(f"Режим выборки не поддерживается для данной головы (head) {model.cfg.head.lower()}!")

    return samples


def _subset_target(parameter: Dict[str, torch.Tensor], n_target: int, steps: int) -> Dict[str, torch.Tensor]:
    # определить, какие нейроны выхода соответствуют n-му целевому признаку
    start = n_target * steps
    end = (n_target + 1) * steps
    parameter_sub = parameter[:, :, start:end]

    return parameter_sub


def _handle_negative_values(cfg: Config, values: torch.Tensor, sample_values: Callable,
                            scaler: Dict[str, Union[pd.Series, xarray.Dataset]], nth_target: int) -> torch.Tensor:
    """Обработка отрицательных выборок, возникающих при взятии выборок из оценок неопределенности.

    Поддерживаемые режимы: 
    (а) 'clip' для непосредственной обрезки значений на уровне нуля и 
    (б) 'truncate' для повторной выборки значений, которые оказались ниже нуля.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    values : torch.Tensor
        Тензор с выбранными значениями.
    sample_values : Callable
        Функция выборки, позволяющая проводить повторную выборку в случае обработки ограничения (truncation).
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Нормализатор запускаемого процесса.
    nth_target : int
        Индекс выбранного целевого признака в списке cfg.target_variables.

    Возвращает
    -------
    torch.Tensor
        Ограниченные значения в соответствии с пользовательскими настройками.  
    """
    center = scaler["xarray_feature_center"][cfg.target_variables[nth_target]]
    scale = scaler["xarray_feature_scale"][cfg.target_variables[nth_target]]
    normalized_zero = -torch.from_numpy((center / scale).values).to(values)
    if cfg.negative_sample_handling.lower() == 'clip':
        values = torch.clamp(values, min=normalized_zero)
    elif cfg.negative_sample_handling.lower() == 'truncate':
        values_smaller_zero = values < normalized_zero
        try_count = 0
        while torch.any(values_smaller_zero.flatten()):
            values[values_smaller_zero] = sample_values(values_smaller_zero)
            values_smaller_zero = values < normalized_zero
            try_count += 1
            if try_count >= cfg.negative_sample_max_retries:
                break
    elif cfg.negative_sample_handling is None or cfg.negative_sample_handling.lower() == 'none':
        pass
    else:
        raise NotImplementedError(
            f"Вариант {cfg.negative_sample_handling} не поддерживается для обработки отрицательных выборок!")

    return values


def _sample_gaussian_mixtures(ids: List[int], m_sub: torch.Tensor, s_sub: torch.Tensor,
                              p_sub: torch.Tensor) -> torch.Tensor:
    # неограниченная выборка:
    categorical = Categorical(p_sub)
    pis = categorical.sample().data
    mask_gmm = torch.zeros(p_sub.shape, dtype=torch.bool) \
        .to(device=p_sub.device) \
        .scatter_(2, pis.unsqueeze(2), True)

    # идентификаторы используются для локально-зависимой повторной выборки при режиме 'truncation' в методе '_handle_negative_values'
    values = s_sub \
        .data.new(s_sub[ids][mask_gmm[ids]].shape[0]) \
        .normal_() \
        .flatten() \
        .mul(s_sub[ids][mask_gmm[ids]]) \
        .add(m_sub[ids][mask_gmm[ids]])
    return values


def _sample_asymmetric_laplacians(ids: List[int], m_sub: torch.Tensor, b_sub: torch.Tensor,
                                  t_sub: torch.Tensor) -> torch.Tensor:
    # идентификаторы используются для локальной повторной выборки при обработке режима 'truncation' в методе '_handle_negative_values'
    prob = torch.FloatTensor(m_sub[ids].shape) \
        .uniform_(0, 1) \
        .to(m_sub.device)  # равномерно выбирать значения между нулём и единицей
    values = torch.where(
        prob < t_sub[ids],  # должно соответствовать выбранной функции потерь
        m_sub[ids] + ((b_sub[ids] * torch.log(prob / t_sub[ids])) / (1 - t_sub[ids])),
        m_sub[ids] - ((b_sub[ids] * torch.log((1 - prob) / (1 - t_sub[ids]))) / t_sub[ids]))
    return values.flatten()


class _SamplingSetup():

    def __init__(self, model: 'BaseModel', data: Dict[str, torch.Tensor], head: str):
        # выполнить проверку модели:
        cfg = model.cfg
        if not cfg.head.lower() == head.lower():
            raise NotImplementedError(f"Выборка для головы {head} не поддерживается для головы {cfg.head}!")

        dropout_modules = [model.dropout.p]

        # Некоторые модели не имеют слоев встраивания (embedding_net)
        implied_statics_embedding, implied_dynamics_embedding = None, None
        if hasattr(model, 'forecast_embedding_net'):
            implied_forecast_statics_embedding = model.forecast_embedding_net.statics_embedding_p_dropout
            implied_forecast_dynamics_embedding = model.forecast_embedding_net.dynamics_embedding_p_dropout
            dropout_modules += [implied_forecast_statics_embedding, implied_forecast_dynamics_embedding]
        if hasattr(model, 'hindcast_embedding_net'):
            implied_hindcast_statics_embedding = model.hindcast_embedding_net.statics_embedding_p_dropout
            implied_hindcast_dynamics_embedding = model.hindcast_embedding_net.dynamics_embedding_p_dropout
            dropout_modules += [implied_hindcast_statics_embedding, implied_hindcast_dynamics_embedding]
        if hasattr(model, 'embedding_net'):
            implied_statics_embedding = model.embedding_net.statics_embedding_p_dropout
            implied_dynamics_embedding = model.embedding_net.dynamics_embedding_p_dropout
            dropout_modules += [implied_statics_embedding, implied_dynamics_embedding]
        # учитывать transformer
        implied_transformer_dropout = None
        if cfg.model.lower() == 'transfomer':
            implied_transformer_dropout = cfg.transformer_dropout
            dropout_modules.append(implied_transformer_dropout)

        max_implied_dropout = max(dropout_modules)
        # проверить нижнюю границу dropout:
        if cfg.mc_dropout and max_implied_dropout <= 0.0:
            raise RuntimeError(f"""{cfg.model} при включённом `mc_dropout` требуется задать коэффициент dropout больше 0.0
                               Текущие подразумеваемые коэффициенты dropout:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - statics_forecast_embedding: {implied_forecast_statics_embedding}
                                  - dynamics_forecast_embedding: {implied_forecast_dynamics_embedding}
                                  - statics_hindcast_embedding: {implied_hindcast_statics_embedding}
                                  - dynamics_hindcast_embedding: {implied_hindcast_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")
        # проверить верхнюю границу dropout:
        if cfg.mc_dropout and max_implied_dropout >= 1.0:
            raise RuntimeError(f"""Максимальное значение коэффициента dropout - 1. Проверьте настройки dropout:
                               Текущие подразумеваемые коэффициенты dropout:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - statics_forecast_embedding: {implied_forecast_statics_embedding}
                                  - dynamics_forecast_embedding: {implied_forecast_dynamics_embedding}
                                  - statics_hindcast_embedding: {implied_hindcast_statics_embedding}
                                  - dynamics_hindcast_embedding: {implied_hindcast_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")

        # назначить свойства настройки:
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.number_of_targets = len(cfg.target_variables)
        self.mc_dropout = cfg.mc_dropout
        self.predict_last_n = cfg.predict_last_n

        # определить подходящий суффикс частоты:
        if len(self.cfg.use_frequencies) > 1:
            self.freq_suffixes = [f'_{freq}' for freq in cfg.use_frequencies]
        else:
            self.freq_suffixes = ['']

        self.batch_size_data = data[f'y{self.freq_suffixes[0]}'].shape[0]


def _get_frequency_last_n(predict_last_n: dict[str, int] | int, freq_suffix: str, use_frequencies: list[str]):
    if isinstance(predict_last_n, int):
        frequency_last_n = predict_last_n
    else:
        if freq_suffix != '':
            frequency_last_n = predict_last_n[freq_suffix[1:]]
        else:
            frequency_last_n = predict_last_n[use_frequencies[0]]
    return frequency_last_n


def sample_mcd(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int,
               scaler: Dict[str, Union[pd.Series, xarray.Dataset]]) -> Dict[str, torch.Tensor]:
    """Выборка точечных прогнозов на основе MC-Dropout.

    Простая выборка. Эта функция выполняет `n_samples` прямых проходов для каждого примера в пакете. В настоящее время 
    она полезна только для моделей с дропаутом, чтобы выполнять выборку методом MC-Dropout. 
    Обратить внимание: вызов этой функции принудительно переведёт модель в режим обучения (`model.train()`) и 
    не вернёт её в предыдущее состояние.

    Обработка отрицательных выборок в настоящее время поддерживает 
    (а) 'clip' для прямой обрезки точек выборки на уровне нуля и
    (б) 'truncate' для повторной выборки точек, которые находятся ниже нуля.
    Режим задаётся аргументом конфигурации 'negative_sample_handling'.

    Параметры
    ----------
    model : BaseModel
        Модель с непробабилистической головой.
    data : Dict[str, torch.Tensor]
        Словарь, содержащий входные признаки в виде пар ключ-значение.
    n_samples : int
        Количество выборок, которые необходимо сгенерировать для каждого входного примера.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Нормализатор запуска.

    Возвращает
    -------
    Dict[str, torch.Tensor]
        Словарь, содержащий выборочные выходные данные модели для последних `predict_last_n` (аргумент конфигурации) 
        временных шагов каждой частоты.
    """
    setup = _SamplingSetup(model, data, model.cfg.head)

    # принудительно перевести модель в режим обучения для mc_dropout:
    if setup.mc_dropout:
        model.train()

    # взятие выборок для различных частот и целевых признаков:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        sample_points = []
        frequency_last_n = _get_frequency_last_n(setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies)

        x_d = data[f'x_d{freq_suffix}']
        some_key = list(x_d)[0]
        ids = list(range(x_d[some_key].shape[0]))

        for nth_target in range(setup.number_of_targets):
            # неограниченная выборка:
            def _sample_values(ids: List[int]) -> torch.Tensor:
                # идентификаторы используются для локализации выборки при обработке 'truncation' в методе `_handle_negative_values`
                target_values = torch.zeros(len(ids), frequency_last_n, n_samples)
                for i in range(n_samples):  # Выполнять прямые проходы отдельно для каждой частоты, чтобы гарантировать независимость
                    prediction = model(data)
                    value_buffer = prediction[f'y_hat{freq_suffix}'][:, -frequency_last_n:, 0]
                    target_values[ids, -frequency_last_n:, i] = value_buffer.detach().cpu()
                return target_values

            values = _sample_values(ids)

            # связать значения и добавить их в `sample_points`:
            values = _handle_negative_values(setup.cfg, values, _sample_values, scaler, nth_target)
            sample_points.append(values)

        # добавить `sample_points` в словарь выборок:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: torch.stack(sample_points, 2)})

    return samples


def sample_gmm(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int,
               scaler: Dict[str, Union[pd.Series, xarray.Dataset]]) -> Dict[str, torch.Tensor]:
    """Взятие выборок точечных прогнозов с использованием головы на основе смеси Гауссианов (GMM).

    Функция генерирует `n_samples` выборочных точек GMM для каждого элемента батча. Конкретно, модель исполняется 
    однократно (прямой проход), а затем выборочные точки создаются путём выборки из результирующей смеси.
    Хорошие справочники по изучению GMM представлены в работах [#]_ и [#]_. 

    Обработка отрицательных выборок в настоящее время поддерживает 
    (а) 'clip' для прямой обрезки выборочных точек на уровне нуля и
    (б) 'truncate' для повторной выборки точек, которые находятся ниже нуля.
    Режим определяется аргументом конфигурации 'negative_sample_handling'.
     
    Примечание: если в конфигурации установлена настройка 'mc_dropout' в значение true, данная функция 
    принудительно переведёт модель в режим обучения (`model.train()`) и не восстановит её исходное состояние.

    Параметры
    ----------
    model : BaseModel
        Модель с головой GMM.
    data : Dict[str, torch.Tensor]
        Словарь, содержащий входные признаки в виде пар ключ-значение.
    n_samples : int
        Количество выборок, которые необходимо сгенерировать для каждого входного примера.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Нормализатор запущенного процесса.

    Возвращает
    -------
    Dict[str, torch.Tensor]
        Словарь, содержащий выборочные выходные данные модели для последних `predict_last_n` (аргумент конфигурации) 
        временных шагов каждой частоты.

    References
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """
    setup = _SamplingSetup(model, data, "gmm")

    # принудительно перевести модель в режим обучения, если включен mc_dropout::
    if setup.mc_dropout:
        model.train()

    # сделать прогноз:
    pred = model(data)

    # производить выборку для различных частот:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # получить predict_last_n выцбранного режима:
        frequency_last_n = _get_frequency_last_n(setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies)

        # инициализировать тензор sample_points для выборки:
        sample_points = torch.zeros((setup.batch_size_data, frequency_last_n, setup.number_of_targets, n_samples))
        sample_points *= torch.tensor(float('nan'))  # установить начальные значения sample_points как NaN

        # Смесь Гауссианов (GMM) состоит из трёх частей: средние значения (m/mu), дисперсии (s/sigma), и веса (p/pi):
        m, s, p = pred[f'mu{freq_suffix}'], \
                  pred[f'sigma{freq_suffix}'], \
                  pred[f'pi{freq_suffix}']

        for nth_target in range(setup.number_of_targets):
            m_target = _subset_target(m[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            s_target = _subset_target(s[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            p_target = _subset_target(p[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)

            mask_nan = ~torch.isnan(m_target[:, -1, 0])
            if any(mask_nan):  # пропускать обработку мини-пакета целиком, если он недействителен.
                m_sub = torch.repeat_interleave(m_target[mask_nan, :, :], n_samples, dim=0)
                s_sub = torch.repeat_interleave(s_target[mask_nan, :, :], n_samples, dim=0)
                p_sub = torch.repeat_interleave(p_target[mask_nan, :, :], n_samples, dim=0)

                # взять выборки значений, обработать отрицательные и добавить их в выборочные точки:
                values = _sample_gaussian_mixtures(torch.ones(s_sub.shape, dtype=bool), m_sub, s_sub, p_sub)
                values = _handle_negative_values(
                    setup.cfg,
                    values,
                    sample_values=lambda ids: _sample_gaussian_mixtures(ids, m_sub, s_sub, p_sub),
                    scaler=scaler,
                    nth_target=nth_target)
                values = values.view(-1, n_samples, frequency_last_n).permute(0, 2, 1)

                sample_points[mask_nan, :, nth_target, :] = values.detach().cpu()

        # добавить `sample_points` в словарь выборок:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: sample_points})
    return samples


def sample_cmal(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int,
                scaler: Dict[str, Union[pd.Series, xarray.Dataset]]) -> Dict[str, torch.Tensor]:
    """Создание точечных прогнозов с использованием головы на основе счетной смеси асимметричных Лапласа (CMAL).

    Функция генерирует `n_samples` выборочных точек CMAL для каждого элемента батча. Фактически, модель
    выполняется единожды (прямой проход), а затем выборочные точки формируются посредством выборки из полученной смеси.
    Основные сведения о CMAL можно найти в источнике [#]_.

    В настоящее время обработка отрицательных выборок поддерживает
    (а) 'clip' для прямой обрезки выборочных точек на уровне нуля и 
    (б) 'truncate' для повторной выборки точек, которые ниже нуля.
    Режим обработки можно задать с помощью аргумента конфигурации 'negative_sample_handling'.

    Примечание: если в конфигурации включена настройка 'mc_dropout', эта функция принудительно 
    переведёт модель в режим обучения (`model.train()`) и не возвратит её в исходное состояние.

    Параметры
    ----------
    model : BaseModel
        Модель с головой CMAL.
    data : Dict[str, torch.Tensor]
        Словарь, содержащий входные признаки в виде пар ключ-значение.
    n_samples : int
        Количество выборок, которые необходимо сформировать для каждого входного примера.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Нормализатор запущенного процесса.

    Возвращает
    -------
    Dict[str, torch.Tensor]
        Словарь, содержащий выборочные выходные данные модели для последних `predict_last_n` (аргумент конфигурации) 
        временных шагов каждой частоты. Форма выходного тензора для каждой частоты -
        ``[batch size, predict_last_n, n_samples]``.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing: 
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295,
        2020.
    """
    setup = _SamplingSetup(model, data, "cmal")

    # принудительно перевести модель в режим обучения, если включён mc_dropout
    if setup.mc_dropout:
        model.train()

    # сделать прогноз:
    pred = model(data)

    # выборка для различных частот:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # получение predict_last_n для заданного режима:
        frequency_last_n = _get_frequency_last_n(setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies)

        # CMAL состоит из четырёх частей:  средние значения (m/mu), масштабы  (b), асимметрии (t/) и веса (p/pi):
        m = pred[f'mu{freq_suffix}']
        b = pred[f'b{freq_suffix}']
        t = pred[f'tau{freq_suffix}']
        p = pred[f'pi{freq_suffix}']

        sample_points = []
        for nth_target in range(setup.number_of_targets):
            # предустановки для выборки:
            m_target = _subset_target(m[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            b_target = _subset_target(b[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            t_target = _subset_target(t[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            p_target = _subset_target(p[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)

            m_target = torch.repeat_interleave(m_target, n_samples, dim=0)
            b_target = torch.repeat_interleave(b_target, n_samples, dim=0)
            t_target = torch.repeat_interleave(t_target, n_samples, dim=0)
            p_target = torch.repeat_interleave(p_target, n_samples, dim=0)

            # процедура выборки:
            values = torch.zeros((setup.batch_size_data * n_samples, frequency_last_n)).to(setup.device)
            values *= torch.tensor(float('nan'))  # установить целевые sample_points в NaN 
            for nth_timestep in range(frequency_last_n):

                mask_nan = ~torch.isnan(p_target[:, nth_timestep, 0])
                if any(mask_nan):  # пропустить, если весь мини-батч недействителен
                    sub_choices = torch.multinomial(p_target[mask_nan, nth_timestep, :], num_samples=1)
                    t_sub = t_target[mask_nan, nth_timestep, :].gather(1, sub_choices)
                    m_sub = m_target[mask_nan, nth_timestep, :].gather(1, sub_choices)
                    b_sub = b_target[mask_nan, nth_timestep, :].gather(1, sub_choices)

                    ids = torch.ones(b_sub.shape, dtype=bool)
                    values_unbound = _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)
                    values[mask_nan, nth_timestep] = _handle_negative_values(
                        setup.cfg,
                        values_unbound,
                        sample_values=lambda ids: _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub),
                        scaler=scaler,
                        nth_target=nth_target)

            # добавить значения в sample_points:
            values = values.permute(1, 0).reshape(frequency_last_n, -1, n_samples).permute(1, 0, 2)
            values = values.detach().cpu()
            sample_points.append(values)

        # добавить sample_points в словарь выборок:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: torch.stack(sample_points, 2)})
    return samples


def sample_umal(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int,
                scaler: Dict[str, Union[pd.Series, xarray.Dataset]]) -> Dict[str, torch.Tensor]:
    """Sample point predictions with the Uncountable Mixture of Asymmetric Laplacians (UMAL) head.

    This function generates `n_samples` UMAL sample points for each entry in the batch. Concretely, the model is 
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
    Details about the UMAL approach can be found in [#]_.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b) 
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument 
    'negative_sample_handling'.
    
    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`) 
    and not set it back to its original state. 

    Parameters
    ----------
    model : BaseModel
        A model with an UMAL head.
    data : Dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        Number of samples to generate for each input sample.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Scaler of the run.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
        each frequency.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """
    setup = _SamplingSetup(model, data, "umal")

    # force model into train mode if mc_dropout:
    if setup.mc_dropout:
        model.train()

    # n_taus expands the batch by itself and adds a sampled tau as input (new_batch_size = n_taus*batch_size):
    if 'y_extended' not in data:
        data = model.pre_model_hook(data, is_train=False)

    # make predictions:
    pred = model(data)

    # sample:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # get predict_last_n for the given the mode:
        frequency_last_n = _get_frequency_last_n(setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies)

        # UMAL has 2 parts: means (m/mu), scales (b); the tau is randomly chosen:
        m = pred[f'mu{freq_suffix}']
        b = pred[f'b{freq_suffix}']
        t = data[f'tau{freq_suffix}']

        # sampling presets:
        m_wide = torch.cat(m[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)
        b_wide = torch.cat(b[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)

        # for now we just use a single tau for all targets:
        t_target = torch.cat(t[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)

        # sample over targets:
        sample_points = torch.zeros((setup.batch_size_data, frequency_last_n, setup.number_of_targets, n_samples))
        sample_points *= torch.tensor(float('nan'))  # set initial sample_points to nan
        for nth_target in range(setup.number_of_targets):
            # sampling presets:
            m_target = _subset_target(m_wide[:, -frequency_last_n:, :], nth_target, setup.cfg.n_taus)
            b_target = _subset_target(b_wide[:, -frequency_last_n:, :], nth_target, setup.cfg.n_taus)

            # sample over n_samples:
            for nth_sample in range(n_samples):
                sub_choice = np.random.randint(0, setup.cfg.n_taus)

                mask_nan = ~torch.isnan(m_target[:, 0, 0])
                if any(mask_nan):  # skip computation if entire mini-batch is invalid
                    m_sub = m_target[mask_nan, :, sub_choice]
                    b_sub = b_target[mask_nan, :, sub_choice]
                    t_sub = t_target[mask_nan, :, sub_choice]

                    ids = torch.ones(b_sub.shape, dtype=bool)
                    values_unbound = _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)
                    values = _handle_negative_values(
                        setup.cfg,
                        values_unbound,
                        sample_values=lambda ids: _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub),
                        scaler=scaler,
                        nth_target=nth_target)

                    # add values to sample_points:
                    values = values.detach().cpu().unsqueeze(1)
                    sample_points[mask_nan, -frequency_last_n:, nth_target, nth_sample] = values

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: sample_points})
    return samples

def umal_extend_batch(data: Dict[str, torch.Tensor], cfg: Config, n_taus: int = 1, extend_y: bool = False) \
        -> Dict[str, torch.Tensor]:
    """This function extends the batch for the usage in UMAL (see: [#]_). 
    
    UMAL makes an MC approximation to a mixture integral by sampling random asymmetry parameters (tau). This can be 
    parallelized by expanding the batch for each tau.  
    
    Parameters
    ----------
    data : Dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    cfg : Config
        The run configuration.
    n_taus : int
        Number of taus to expand the batch.
    extend_y : bool
        Option to also extend the labels/y. 
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary, containing expanded input features and tau samples as key-value pairs.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """
    # setup:
    if len(cfg.use_frequencies) > 1:
        freq_suffixes = [f'_{freq}' for freq in cfg.use_frequencies]
    else:
        freq_suffixes = ['']

    for freq_suffix in freq_suffixes:
        x_d = data[f'x_d{freq_suffix}']
        # All entries in x_d have the same batch size and sequence length, so we just pick the first.
        some_dynamic_input = x_d[list(x_d)[0]]
        batch_size, seq_length, _ = some_dynamic_input.shape

        predict_last_n = _get_frequency_last_n(cfg.predict_last_n, freq_suffix, cfg.use_frequencies)

        # sample tau within [tau_down, tau_up] and add to data:
        tau = (cfg.tau_up - cfg.tau_down) * torch.rand(batch_size * n_taus, 1, 1) + cfg.tau_down
        tau = tau.repeat(1, seq_length, 1)  # in our convention tau remains the same over all inputs
        tau = tau.to(some_dynamic_input.device)
        data[f'tau{freq_suffix}'] = tau[:, -predict_last_n:, :]

        # extend dynamic inputs with tau and expand batch:
        for k, v in x_d.items():
            x_d[k] = v.repeat(n_taus, 1, 1)
        # prefix with underscore to avoid name clashes with normal input features.
        assert '_tau' not in x_d
        x_d['_tau'] = tau
        data[f'y_extended{freq_suffix}'] = data[f'y{freq_suffix}'].repeat(n_taus, 1, 1)
        if f'x_s{freq_suffix}' in data:
            data[f'x_s{freq_suffix}'] = data[f'x_s{freq_suffix}'].repeat(n_taus, 1)
        if f'x_one_hot{freq_suffix}' in data:
            data[f'x_one_hot{freq_suffix}'] = data[f'x_one_hot{freq_suffix}'].repeat(n_taus, 1)

    return data


@njit
def bernoulli_subseries_sampler(
    data: np.ndarray,
    missing_fraction: float,
    mean_missing_length: float,
    start_sampling_on: bool = True,
) -> np.ndarray:
    """Samples a timeseries according to a pair of Bernoulli processes.

    The objective is to sample subsequences of a given timeseries under two criteria:
        1)  Expected long-term sample ratio (i.e., the total fraction of points in a time series 
            that are not sampled): `missing_fraction`.
        2)  Expected length of continuous subsequences sampled from the timeseries:
            `mean_missing_length`.
    
    This is done by sampling two Bernoulli processes with different rate parameters. One
    process samples on-shifts and one process samples off-shifts. An 'on-shift' occurs
    when the state of the sampler transitions from 'off' (not sampling) to 'on' (sampling),
    and vice-versa. The rate parameters for the on-shift and off-shift processes are
    derived from the input parameters explained above.
 
    Parameters
    ----------
    data : np.ndarray
        Time series data to be sampled. Must be (N, 1) where N is the length of the timeseries.
    missing_fraction : float
        Expected total fraction of points in a time series that are not sampled.
    mean_missing_length : float
        Expected length of continuous subsequences of un-sampled data from the timeseries.
    start_sampling_on: bool
        Whether to start with the sampler turned "on" (True) or "off" (False) at the first
        timestep of the timeseries.

    Returns
    -------
    A copy of the timeseries with NaN's replacing elements that were not sampled.
    """
    # Check if sampling ratio is one or zero. Avoids a divide-by-zero error.
    if missing_fraction == 0:
        return data
    if missing_fraction == 1:
        return np.full(data.shape, np.nan)

    # Check that the input data is a 1-d timeseries.
    if not (data.ndim == 1 or (data.ndim == 2 and data.shape[-1] == 1)):
        raise ValueError('Shape of timeseries data must be N or (N, 1).')

    # Check that the distribution parameters make sense.
    if mean_missing_length < missing_fraction / (1 - missing_fraction):
        raise ValueError('Incompatible distribution parameters in timeseries sampling. Must be: ',
                         'mean_missing_length >= missing_fraction / (1-missing_fraction).')
    if missing_fraction < 0 or missing_fraction > 1:
        raise ValueError('Missing fraction must be in [0,1]')

    if mean_missing_length <= 0:
        raise ValueError('Mean missing length must be > 0.')

    # Derive Bernoulli rate parameters.
    on_shift_rate = 1 / mean_missing_length
    off_shift_rate = on_shift_rate * missing_fraction / (1 - missing_fraction)

    # Initialize storage for the samples.
    sampled_data = np.full(data.shape, np.nan)
    sampled_data[0] = data[0]
    if not start_sampling_on:
        sampled_data[0] = np.nan

    # Bernoulli sampling.
    up_switches = np.random.binomial(n=1, p=on_shift_rate, size=data.shape)
    down_switches = np.random.binomial(n=1, p=off_shift_rate, size=data.shape)

    # Sampling -> stochastic process.
    for n in range(1, len(data)):
        if np.isnan(sampled_data[n - 1]):
            if up_switches[n]:
                sampled_data[n] = data[n]
        else:
            if not down_switches[n]:
                sampled_data[n] = data[n]

    return sampled_data
