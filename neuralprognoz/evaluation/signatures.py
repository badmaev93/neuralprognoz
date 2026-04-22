import warnings
from datetime import datetime
from typing import Dict, List, Tuple

from dateutil.relativedelta import relativedelta

import numpy as np
from numba import njit
from xarray.core.dataarray import DataArray

from neuralprognoz.datautils import utils


def get_available_signatures() -> List[str]:
    """Возвращает список доступных гидрологических индексов.

    Возвращает
    -------
    List[str]
        Список всех доступных гидрологических индексов.
    """
    signatures = [
        "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "q95", "q5", "q_mean", "hfd_mean",
        "baseflow_index", "slope_fdc", "stream_elas", "runoff_ratio"
    ]
    return signatures


def calculate_all_signatures(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> Dict[str, float]:
    """Вычисляет все гидрологические индексы с заданными значениями по умолчанию.

    Параметры
    ----------
    da : DataArray
        Массив значений стока, для которого будут вычислены гидрологические индексы.
    prcp : DataArray
        Массив значений осадков.
    datetime_coord : str, optional
        Координата даты и времени в переданном DataArray. Попытка автоматического определения, если не указана.

    Возвращает
    -------
    Dict[str, float]
        Словарь с названиями гидрологических индексов в качестве ключей и их значениями в качестве значений.
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    results = {
        "high_q_freq": high_q_freq(da, datetime_coord=datetime_coord),
        "high_q_dur": high_q_dur(da),
        "low_q_freq": low_q_freq(da, datetime_coord=datetime_coord),
        "low_q_dur": low_q_dur(da),
        "zero_q_freq": zero_q_freq(da),
        "q95": q95(da),
        "q5": q5(da),
        "q_mean": q_mean(da),
        "hfd_mean": hfd_mean(da, datetime_coord=datetime_coord),
        "baseflow_index": baseflow_index(da, datetime_coord=datetime_coord)[0],
        "slope_fdc": slope_fdc(da),
        "stream_elas": stream_elas(da, prcp, datetime_coord=datetime_coord),
        "runoff_ratio": runoff_ratio(da, prcp, datetime_coord=datetime_coord)
    }
    return results


def calculate_signatures(da: DataArray,
                         signatures: List[str],
                         datetime_coord: str = None,
                         prcp: DataArray = None) -> Dict[str, float]:
    """Вычисляет указанные гидрологические индексы с заданными значениями по умолчанию.

    Параметры
    ----------
    da : DataArray
        Массив значений стока, для которого будут вычислены гидрологические индексы.
    signatures : List[str]
        Список имен гидрологических индексов для вычисления.
    datetime_coord : str, optional
        Координата даты и времени в переданном DataArray. Попытка автоматического определения, если не указана.
    prcp : DataArray, optional
        Массив значений осадков. Требуется для гидрологических индексов 'runoff_ratio' и 'streamflow_elas'.

    Возвращает
    -------
    Dict[str, float]
        Словарь с названиями гидрологических индексов в качестве ключей и их значениями в качестве значений.

    Вызывает
    ------
    ValueError
        Если переданное имя гидрологического индекса не существует.
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    values = {}
    for signature in signatures:
        if signature == "high_q_freq":
            values["high_q_freq"] = high_q_freq(da, datetime_coord=datetime_coord)
        elif signature == "high_q_dur":
            values["high_q_dur"] = high_q_dur(da)
        elif signature == "low_q_freq":
            values["low_q_freq"] = low_q_freq(da, datetime_coord=datetime_coord)
        elif signature == "low_q_dur":
            values["low_q_dur"] = low_q_dur(da)
        elif signature == "zero_q_freq":
            values["zero_q_freq"] = zero_q_freq(da)
        elif signature == "q95":
            values["q95"] = q95(da)
        elif signature == "q5":
            values["q5"] = q5(da)
        elif signature == "q_mean":
            values["q_mean"] = q_mean(da)
        elif signature == "hfd_mean":
            values["hfd_mean"] = hfd_mean(da, datetime_coord=datetime_coord)
        elif signature == "baseflow_index":
            values["baseflow_index"] = baseflow_index(da, datetime_coord=datetime_coord)[0]
        elif signature == "slope_fdc":
            values["slope_fdc"] = slope_fdc(da)
        elif signature == "runoff_ratio":
            values["runoff_ratio"] = runoff_ratio(da, prcp, datetime_coord=datetime_coord)
        elif signature == "stream_elas":
            values["stream_elas"] = stream_elas(da, prcp, datetime_coord=datetime_coord)
        else:
            ValueError(f"Неизвестный гидрологический индекс {signature}")
    return values


@njit
def _split_list(alist: list, min_length: int = 0) -> list:
    """Разбивает список индексов на списки последовательных индексов длиной не менее `min_length`. """
    newlist = []
    start = 0
    for index, value in enumerate(alist):
        if index < len(alist) - 1:
            if alist[index + 1] > value + 1:
                end = index + 1
                if end - start >= min_length:
                    newlist.append(alist[start:end])
                start = end
        else:
            if len(alist) - start >= min_length:
                newlist.append(alist[start:len(alist)])
    return newlist


def high_q_dur(da: DataArray, threshold: float = 9.) -> float:
    """Вычисляет продолжительность паводков.

    Средняя продолжительность паводков (количество последовательных шагов >`threshold` раз медианного расхода) [#]_,
    [#]_ (Таблица 2).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    threshold : float, optional
        Порог для определения высоких расходов. Значения, превышающие ``threshold * median``, считаются высокими расходами.

    Возвращает
    -------
    float
        Продолжительность паводков

    Ссылки
    ----------
    .. [#] Clausen, B. and Biggs, B. J. F.: Flow variables for ecological studies in temperate streams: groupings based
        on covariance. Journal of Hydrology, 2000, 237, 184--197, doi:10.1016/S0022-1694(00)00306-1
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    median_flow = float(da.median())
    idx = np.where(da.values > threshold * median_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        hqd = np.mean([len(p) for p in periods])
    else:
        hqd = np.nan
    return hqd


def low_q_dur(da: DataArray, threshold: float = 0.2) -> float:
    """Вычисляет продолжительность межени.

    Средняя продолжительность межени (количество последовательных шагов <`threshold` раз медианного расхода) [#]_,
    [#]_ (Таблица 2).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    threshold : float, optional
        Порог для определения межени. Значения, ниже ``threshold * median``, считаются меженью.

    Возвращает
    -------
    float
        Продолжительность межени

    Ссылки
    ----------
    .. [#] Olden, J. D. and Poff, N. L.: Redundancy and the choice of hydrologic indices for characterizing streamflow
        regimes. River Research and Applications, 2003, 19, 101--121, doi:10.1002/rra.700
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    mean_flow = float(da.mean())
    idx = np.where(da.values < threshold * mean_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        lqd = np.mean([len(p) for p in periods])
    else:
        lqd = np.nan
    return lqd


def zero_q_freq(da: DataArray) -> float:
    """Вычисляет частоту нулевых расходов.

    Частота шагов с нулевым расходом.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.

    Возвращает
    -------
    float
        Частота нулевых расходов.
    """
    # количество шагов с нулевым расходом
    n_steps = (da == 0).sum()

    return float(n_steps / len(da))


def high_q_freq(da: DataArray, datetime_coord: str = None, threshold: float = 9.) -> float:
    """Вычисляет частоту паводков.

    Частота паводков (>`threshold` раз медианного расхода) [#]_, [#]_ (Таблица 2).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана.
    threshold : float, optional
        Порог для определения высоких расходов. Значения, превышающие ``threshold * median``, считаются высокими расходами.

    Возвращает
    -------
    float
        Частота паводков

    Ссылки
    ----------
    .. [#] Clausen, B. and Biggs, B. J. F.: Flow variables for ecological studies in temperate streams: groupings based
        on covariance. Journal of Hydrology, 2000, 237, 184--197, doi:10.1016/S0022-1694(00)00306-1
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # определение даты первого января в периоде данных
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # конечная дата первого полного года
    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # определение медианного расхода за весь период
    median_flow = da.median(skipna=True)

    hqfs = []
    while end_date < last_date:

        data = da.sel({datetime_coord: slice(start_date, end_date)})

        # количество шагов с расходом выше threshold * median за один год
        n_steps = (data > (threshold * median_flow)).sum()

        hqfs.append(float(n_steps))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(hqfs)


def low_q_freq(da: DataArray, datetime_coord: str = None, threshold: float = 0.2) -> float:
    """Вычисляет частоту межени.

    Частота межени (<`threshold` раз медианного расхода) [#]_, [#]_ (Таблица 2).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана.
    threshold : float, optional
        Порог для определения межени. Значения, ниже ``threshold * median``, считаются меженью.

    Возвращает
    -------
    float
        Частота межени

    Ссылки
    ----------
    .. [#] Olden, J. D. and Poff, N. L.: Redundancy and the choice of hydrologic indices for characterizing streamflow
        regimes. River Research and Applications, 2003, 19, 101--121, doi:10.1002/rra.700
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # определение даты первого января в периоде данных
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # конечная дата первого полного года
    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # определение среднего расхода за весь период
    mean_flow = da.mean(skipna=True)

    lqfs = []
    while end_date < last_date:

        data = da.sel({datetime_coord: slice(start_date, end_date)})

        # количество шагов с расходом ниже threshold * median за один год
        n_steps = (data < (threshold * mean_flow)).sum()

        lqfs.append(float(n_steps))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(lqfs)


def hfd_mean(da: DataArray, datetime_coord: str = None) -> float:
    """Вычисляет среднюю продолжительность периода половодья.

    Средняя продолжительность периода половодья (шаг, на котором кумулятивный сток с 1 октября
    достигает половины годового стока) [#]_.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана.

    Возвращает
    -------
    float
        Средняя продолжительность периода половодья

    Ссылки
    ----------
    .. [#] Court, A.: Measures of streamflow timing. Journal of Geophysical Research (1896-1977), 1962, 67, 4335--4339,
        doi:10.1029/JZ067i011p04335
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # определение даты первого октября в периоде данных
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    doys = []
    while end_date < last_date:

        # вычисление кумулятивной суммы для выбранного периода
        data = da.sel({datetime_coord: slice(start_date, end_date)})
        cs = data.cumsum(skipna=True)

        # поиск шагов с кумулятивным расходом выше половины годовой суммы
        hf_steps = np.where(~np.isnan(cs.where(cs > data.sum(skipna=True) / 2).values))[0]

        # пропуск дней без стока
        if len(hf_steps) > 0:
            # сохранение первого шага в результирующий массив
            doys.append(hf_steps[0])

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(doys)


def q5(da: DataArray) -> float:
    """Вычисляет 5-й квантиль расхода.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.

    Возвращает
    -------
    float
        5-й квантиль расхода.
    """
    return float(da.quantile(0.05))


def q95(da: DataArray) -> float:
    """Вычисляет 95-й квантиль расхода.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.

    Возвращает
    -------
    float
        95-й квантиль расхода.
    """
    return float(da.quantile(0.95))


def q_mean(da: DataArray) -> float:
    """Вычисляет средний расход.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.

    Возвращает
    -------
    float
        Средний расход.
    """
    return float(da.mean())


@njit
def _baseflow_index_jit(streamflow: np.ndarray, alpha: float, warmup: int, n_passes: int) -> Tuple[float, np.ndarray]:
    non_nan_indices = np.where(~np.isnan(streamflow))[0]
    non_nan_streamflow_runs = _split_list(non_nan_indices, min_length=warmup + 1)
    if len(non_nan_streamflow_runs) == 0:
        # нет непрерывной последовательности ненулевых значений, достаточной для расчета базового стока.
        return np.nan, np.full_like(streamflow, np.nan)

    streamflow_sum = 0
    overall_baseflow = np.full_like(streamflow, np.nan)
    for non_nan_run in non_nan_streamflow_runs:
        streamflow_run = streamflow[non_nan_run]

        # зеркальное отражение расхода длиной 'window' в начале и конце
        padded_streamflow = np.zeros((len(streamflow_run) + 2 * warmup))
        padded_streamflow[warmup:-warmup] = streamflow_run
        padded_streamflow[:warmup] = streamflow_run[1:warmup + 1][::-1]
        padded_streamflow[-warmup:] = streamflow_run[-warmup - 1:-1][::-1]

        baseflow = padded_streamflow
        for _ in range(n_passes):
            new_baseflow = np.zeros_like(padded_streamflow)
            quickflow = baseflow[0]
            for i in range(1, len(padded_streamflow)):
                quickflow = alpha * quickflow + (1 + alpha) * (baseflow[i] - baseflow[i - 1]) / 2
                if quickflow > 0:
                    new_baseflow[i] = baseflow[i] - quickflow
                else:
                    new_baseflow[i] = baseflow[i]

            # переключение между прямым и обратным проходами. Входные данные для следующей итерации - это базовый сток, 
            # сгенерированный в этой итерации
            baseflow = np.flip(new_baseflow)

        overall_baseflow[non_nan_run] = baseflow[warmup:-warmup]
        streamflow_sum += np.sum(streamflow_run)

    bf_index = np.nansum(overall_baseflow) / streamflow_sum

    return bf_index, overall_baseflow


def baseflow_index(da: DataArray,
                   alpha: float = 0.98,
                   warmup: int = 30,
                   n_passes: int = None,
                   datetime_coord: str = None) -> Tuple[float, DataArray]:
    """Вычисляет индекс базового стока.

    Отношение среднего базового стока к среднему расходу [#]_. Если `da` содержит значения NaN, базовый сток вычисляется для каждого
    непрерывного сегмента из более чем `warmup` ненулевых значений.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    alpha : float, optional
        Параметр фильтра alpha.
    warmup : int, optional
        Количество шагов прогрева.
    n_passes : int, optional
        Количество проходов (чередующихся прямого и обратного). Должно быть нечетным числом. Если None, будет использовано
        3 для ежедневных и 9 для почасовых данных, и будет выдана ошибка для всех других частот.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана. Используется для определения 
        частоты, если `n_passes` не указан.

    Возвращает
    -------
    Tuple[float, DataArray]
        Индекс базового стока и массив базового стока. Массив базового стока содержит NaN везде, где базовый сток не был
        вычислен из-за NaN в `da`.

    Вызывает
    ------
    ValueError
        Если `da` имеет частоту, отличную от ежедневной или почасовой, и `n_passes` не указан.

    Ссылки
    ----------
    .. [#] Ladson, T. R., Brown, R., Neal, B., and Nathan, R.: A Standard Approach to Baseflow Separation Using The
        Lyne and Hollick Filter. Australasian Journal of Water Resources, Taylor & Francis, 2013, 17, 25--34,
        doi:10.7158/13241583.2013.11465417
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    if n_passes is None:
        freq = utils.infer_frequency(da[datetime_coord].values)
        if freq == '1D':
            n_passes = 3
        elif freq == '1h':
            n_passes = 9
        else:
            raise ValueError(f'Для частот, отличных от ежедневной или почасовой, n_passes должен быть указан.')
    if n_passes % 2 != 1:
        warnings.warn('n_passes должен быть нечетным числом. Возвращаемый базовый сток будет обратным.')

    # вызов jit скомпилированной функции для расчета базового стока
    bf_index, baseflow = _baseflow_index_jit(da.values, alpha, warmup, n_passes)

    # преобразование базового стока в DataArray с использованием координат массива стока
    da_baseflow = da.copy()
    da_baseflow.data = baseflow

    return bf_index, da_baseflow


def slope_fdc(da: DataArray, lower_quantile: float = 0.33, upper_quantile: float = 0.66) -> float:
    """Вычисляет наклон кривой длительности стока.

    Наклон кривой длительности стока (между логарифмированными `lower_quantile` и `upper_quantile`) [#]_ (Ур. 3).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    lower_quantile : float, optional
        Нижний квантиль, используемый при расчете наклона.
    upper_quantile : float, optional
        Верхний квантиль, используемый при расчете наклона.

    Возвращает
    -------
    float
        Наклон кривой длительности стока.

    Ссылки
    ----------
    .. [#] Sawicz, K., Wagener, T., Sivapalan, M., Troch, P. A., and Carrillo, G.: Catchment classification: empirical
        analysis of hydrologic similarity based on catchment function in the eastern USA.
        Hydrology and Earth System Sciences, 2011, 15, 2895--2911, doi:10.5194/hess-15-2895-2011
    """
    # сортировка расхода по убыванию
    fdc = da.sortby(da, ascending=False)

    # получение индексов нижнего и верхнего квантилей
    idx_lower = np.round(lower_quantile * len(fdc)).astype(int)
    idx_upper = np.round(upper_quantile * len(fdc)).astype(int)

    value = (np.log(fdc[idx_lower].values + 1e-8) - np.log(fdc[idx_upper].values + 1e-8)) / (upper_quantile -
                                                                                             lower_quantile)

    return value


def runoff_ratio(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> float:
    """Вычисляет коэффициент стока.

    Коэффициент стока (отношение среднего расхода к средней осадкам) [#]_ (Ур. 2).

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    prcp : DataArray
        Массив значений осадков.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана.

    Возвращает
    -------
    float
        Коэффициент стока.

    Ссылки
    ----------
    .. [#] Sawicz, K., Wagener, T., Sivapalan, M., Troch, P. A., and Carrillo, G.: Catchment classification: empirical
        analysis of hydrologic similarity based on catchment function in the eastern USA.
        Hydrology and Earth System Sciences, 2011, 15, 2895--2911, doi:10.5194/hess-15-2895-2011
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # переименование координаты осадков (во избежание проблем с 'index' или 'date')
    prcp = prcp.rename({list(prcp.coords.keys())[0]: datetime_coord})

    # срез осадков по тому же временному окну, что и расход
    prcp = prcp.sel({datetime_coord: slice(da.coords[datetime_coord][0], da.coords[datetime_coord][-1])})

    # расчет коэффициента стока
    value = da.mean() / prcp.mean()

    return float(value)


def stream_elas(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> float:
    """Вычисляет эластичность стока.

    Эластичность стока по осадкам (чувствительность стока к изменениям осадков в годовом масштабе) [#]_.

    Параметры
    ----------
    da : DataArray
        Массив значений стока.
    prcp : DataArray
        Массив значений осадков.
    datetime_coord : str, optional
        Координата времени в переданном DataArray. Попытка автоматического определения, если не указана.

    Возвращает
    -------
    float
        Эластичность стока.

    Ссылки
    ----------
    .. [#] Sankarasubramanian, A., Vogel, R. M., and Limbrunner, J. F.: Climate elasticity of streamflow in the
        United States. Water Resources Research, 2001, 37, 1771--1781, doi:10.1029/2000WR900330
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # переименование координаты осадков (во избежание проблем с 'index' или 'date')
    prcp = prcp.rename({list(prcp.coords.keys())[0]: datetime_coord})

    # срез осадков по тому же временному окну, что и расход
    prcp = prcp.sel({datetime_coord: slice(da.coords[datetime_coord][0], da.coords[datetime_coord][-1])})

    # определение даты первого октября в периоде данных
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # маскирование только валидных временных шагов (только расход имеет пропущенные значения)
    idx = (da >= 0) & (~da.isnull())
    da = da[idx]
    prcp = prcp[idx]

    # расчет долгосрочных средних
    q_mean_total = da.mean()
    p_mean_total = prcp.mean()

    values = []
    while end_date < last_date:
        q = da.sel({datetime_coord: slice(start_date, end_date)})
        p = prcp.sel({datetime_coord: slice(start_date, end_date)})

        val = (q.mean() - q_mean_total) / (p.mean() - p_mean_total) * (p_mean_total / q_mean_total)
        values.append(val)

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.median([float(v) for v in values])
