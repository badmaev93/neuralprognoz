import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict
import numba

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from neuralprognoz.datautils import pet

LOGGER = logging.getLogger(__name__)


def calculate_dyn_climate_indices(precip: pd.Series,
                                  tmax: pd.Series,
                                  tmin: pd.Series,
                                  pet: pd.Series,
                                  window_length: int) -> pd.DataFrame:
    """
    Рассчитывает динамические климатические индексы в скользящем окне.

    Для каждого временного шага индексы пересчитываются на основе данных
    за последние `window_length` шагов.

    Parameters
    ----------
    precip : pd.Series
        Временной ряд осадков.
    t_max : pd.Series
        Временной ряд максимальной температуры.
    t_min : pd.Series
        Временной ряд минимальной температуры.
    pet : pd.Series
        Временной ряд потенциальной эвапотранспирации.
    window_length : int
        Длина окна для расчета (например, 365 для года).

    Returns
    -------
    pd.DataFrame
        DataFrame с рассчитанными климатическими индексами.
    """
    # Убедимся, что все ряды выровнены по времени
    df = pd.DataFrame({'precip': precip, 'tmax': tmax, 'tmin': tmin, 'pet': pet})

    # Проверка на наличие пропусков, которые могут вызвать проблемы
    if df.isnull().values.any():
        # Заполняем пропуски временно, чтобы numba не падала, но это не лучший способ
        # Лучше убедиться, что на вход подаются данные без пропусков за нужный период
        df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    x = df[['precip', 'tmax', 'tmin', 'pet']].values

    # Вызываем Numba-ускоренную функцию для расчетов
    new_features = _numba_climate_indexes(x, window_length=window_length)

    # Формируем DataFrame с результатами
    # Индекс начинается с (window_length - 1)-го элемента исходных данных
    result_index = precip.index[min(window_length, len(precip)) - 1:]
    
    result_df = pd.DataFrame(
        data=new_features,
        columns=[
            'p_mean_dyn',
            'pet_mean_dyn',
            'aridity_dyn',
            't_mean_dyn',
            'frac_snow_dyn',
            'high_prec_freq_dyn',
            'high_prec_dur_dyn',
            'low_prec_freq_dyn',
            'low_prec_dur_dyn'
        ],
        index=result_index)

    return result_df


@njit
def _numba_climate_indexes(features: np.ndarray, window_length: int) -> np.ndarray:
    # features.shape: [#timesteps, 4], колонки: (precip, tmax, tmin, pet)
    n_samples = features.shape[0]
    window_length = min(n_samples, window_length)
    new_features = np.zeros((n_samples - window_length + 1, 9), dtype=np.float64)

    for i in range(new_features.shape[0]):
        x_window = features[i:i + window_length]

        # Избегаем деления на ноль и других проблем с NaN
        p_mean = np.mean(x_window[:, 0])
        pet_mean = np.mean(x_window[:, 3])
        
        if p_mean > 1e-9:
            aridity = pet_mean / p_mean
        else:
            aridity = np.nan

        t_mean = (np.mean(x_window[:, 1]) + np.mean(x_window[:, 2])) / 2

        # Доля осадков в виде снега
        sum_precip = np.sum(x_window[:, 0])
        if sum_precip > 1e-9:
            mean_temp_window = (x_window[:, 1] + x_window[:, 2]) / 2
            snow_precip = np.sum(x_window[mean_temp_window <= 0, 0])
            frac_snow = snow_precip / sum_precip
        else:
            frac_snow = 0.0

        # Частота и продолжительность экстремальных осадков
        high_prec_threshold = 5 * p_mean
        high_prec_freq = np.sum(x_window[:, 0] >= high_prec_threshold) / window_length
        
        high_prec_indices = np.where(x_window[:, 0] >= high_prec_threshold)[0]
        if len(high_prec_indices) > 0:
            high_prec_dur_groups = _split_list(high_prec_indices)
            high_prec_dur = np.mean(np.array([len(p) for p in high_prec_dur_groups]))
        else:
            high_prec_dur = 0.0 # или np.nan, если нет событий

        # Частота и продолжительность засушливых периодов
        low_prec_threshold = 1.0 # мм/день
        low_prec_freq = np.sum(x_window[:, 0] < low_prec_threshold) / window_length
        
        low_prec_indices = np.where(x_window[:, 0] < low_prec_threshold)[0]
        if len(low_prec_indices) > 0:
            low_prec_dur_groups = _split_list(low_prec_indices)
            low_prec_dur = np.mean(np.array([len(p) for p in low_prec_dur_groups]))
        else:
            low_prec_dur = 0.0 # или np.nan, если нет событий

        new_features[i, 0] = p_mean
        new_features[i, 1] = pet_mean
        new_features[i, 2] = aridity
        new_features[i, 3] = t_mean
        new_features[i, 4] = frac_snow
        new_features[i, 5] = high_prec_freq
        new_features[i, 6] = high_prec_dur
        new_features[i, 7] = low_prec_freq
        new_features[i, 8] = low_prec_dur

    return new_features


@njit
def _split_list(a_list: np.ndarray) -> List[np.ndarray]:
    # Создаем типизированный пустой список, чтобы Numba знала, чего ожидать.
    # numba.typed.List() - это специальная конструкция для таких случаев.
    new_list = numba.typed.List()
    if len(a_list) == 0:
        return new_list
        
    start = 0
    for index in range(len(a_list)):
        if index < len(a_list) - 1:
            if a_list[index + 1] > a_list[index] + 1:
                end = index + 1
                new_list.append(a_list[start:end])
                start = end
        else:
            new_list.append(a_list[start:len(a_list)])
    return new_list