import functools
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray
from pandas.tseries.frequencies import to_offset
from ruamel.yaml import YAML
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# В версии 2.2.0 в Pandas произошел переход от "Y" к "YE" и аналогичным идентификаторам. 
# Этот фрагмент кода проверяет, какой из них является правильным для текущей версии Pandas.
_YE_FREQ = 'YE'
_ME_FREQ = 'ME'
_QE_FREQ = 'QE'
try:
    to_offset(_YE_FREQ)
except ValueError:
    _YE_FREQ = 'Y'
    _ME_FREQ = 'M'
    _QE_FREQ = 'Q'


def load_scaler(run_dir: Path) -> Dict[str, Union[pd.Series, xarray.Dataset]]:
    """Загрузка масштабировщика признаков из директории запуска.

    Проверяет директорию запуска на наличие файла масштабировщика в формате yaml (новый) или pickle (старый).

    Параметры
    ----------
    run_dir: Path
        Директория запуска. Должна содержать папку 'train_data', в которой находится файл 'train_data_scaler'.

    Возвращает
    -------
    Dict[str, Union[pd.Series, xarray.Dataset]]
        Словарь, содержащий масштабировщик признаков для статических и динамических признаков.
    
    Вызывает
    ------
    FileNotFoundError
        Если файл масштабировщика не найден в директории запуска.
    """
    scaler_file = run_dir / "train_data" / "train_data_scaler.yml"

    if scaler_file.is_file():
        # читаем масштабировщик из файла
        with scaler_file.open("r") as fp:
            yaml = YAML(typ="safe")
            scaler_dump = yaml.load(fp)

        # преобразуем масштабировщик в формат, ожидаемый neuralprognoz
        scaler = {}
        for key, value in scaler_dump.items():
            if key in ["attribute_means", "attribute_stds", "camels_attr_means", "camels_attr_stds"]:
                scaler[key] = pd.Series(value)
            elif key in ["xarray_feature_scale", "xarray_feature_center"]:
                # Восстанавливаем xarray объекты из нового формата
                if isinstance(value, dict) and 'dims' in value:
                    # Это DataArray
                    # Координаты уже сохранены как словарь с ключами-измерениями
                    coords = value['coords']
                    scaler[key] = xarray.DataArray(
                        data=np.array(value['data'], dtype=np.float32),
                        dims=value['dims'],
                        coords=coords
                    )
                elif isinstance(value, dict) and all(isinstance(v, dict) and 'dims' in v for v in value.values()):
                    # Это Dataset
                    data_vars = {}
                    for var_name, var_data in value.items():
                        # Координаты уже сохранены как словарь с ключами-измерениями
                        coords = var_data['coords']
                        data_vars[var_name] = xarray.DataArray(
                            data=np.array(var_data['data'], dtype=np.float32),
                            dims=var_data['dims'],
                            coords=coords
                        )
                    scaler[key] = xarray.Dataset(data_vars)
                else:
                    # Попытка использовать старый формат для совместимости
                    try:
                        scaler[key] = xarray.Dataset.from_dict(value).astype(np.float32)
                    except Exception as e:
                        LOGGER.warning(f"Не удалось восстановить scaler для {key}: {e}")
                        # Создаем пустой DataArray как fallback
                        scaler[key] = xarray.DataArray(np.array([1.0], dtype=np.float32))

        return scaler

    else:
        scaler_file = run_dir / "train_data" / "train_data_scaler.p"

        if scaler_file.is_file():
            with scaler_file.open('rb') as fp:
                scaler = pickle.load(fp)
            return scaler
        else:
            raise FileNotFoundError(f"Файл масштабирования не найден {scaler_file.parent}. "
                                    "Искали (новый) yaml файл или (старый) pickle файл")


def load_hydroatlas_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Загрузка атрибутов HydroATLAS в pandas DataFrame

    Параметры
    ----------
    data_dir : Path
        Путь к корневой директории набора данных. Должна содержать папку 'hydroatlas_attributes' с файлом
        'attributes.csv'. Файл атрибутов должен иметь столбец 'basin_id'.
    basins : List[str], optional
        Если передано, возвращает только атрибуты для указанных бассейнов. В противном случае возвращаются атрибуты всех бассейнов.

    Возвращает
    -------
    pd.DataFrame
        DataFrame с атрибутами HydroATLAS, индексированный по бассейнам.
    """
    attribute_file = data_dir / "hydroatlas_attributes" / "attributes.csv"
    if not attribute_file.is_file():
        raise FileNotFoundError(attribute_file)

    df = pd.read_csv(attribute_file, dtype={'basin_id': str})
    df = df.set_index('basin_id')

    if basins:
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df


def load_basin_file(basin_file: Path) -> List[str]:
    """Загрузка списка бассейнов из текстового файла.
    
    Примечание: Имена бассейнов не должны оканчиваться на '_period*'
    
    Параметры
    ----------
    basin_file : Path
        Путь к текстовому файлу с бассейнами. Файл должен содержать один идентификатор бассейна на строку, пустые строки игнорируются.

    Возвращает
    -------
    List[str]
        Список идентификаторов бассейнов в виде строк.
        
    Вызывает
    ------
    ValueError
        В случае недопустимых имен бассейнов, которые могут вызвать проблемы.
    """
    with basin_file.open('r') as fp:
        basins = sorted(basin.strip() for basin in fp if basin.strip())

    # проверка имен бассейнов
    problematic_basins = [basin for basin in basins if basin.split('_')[-1].startswith('period')]
    if problematic_basins:
        msg = [
            f"Следующие имена бассейнов недопустимы: {problematic_basins}. Проверьте документацию ",
            "функции 'load_basin_file()' для получения подробной информации."
        ]
        raise ValueError(" ".join(msg))

    return basins


def attributes_sanity_check(df: pd.DataFrame):
    """Вспомогательная функция для проверки пригодности атрибутов для обучения модели.
    
    Эта вспомогательная функция может использоваться для проверки того, имеет ли какой-либо атрибут стандартное отклонение, равное нулю. Это приведет к 
    NaN's при нормализации признаков и, следовательно, к NaN's при обучении модели. Она также проверяет, содержит ли какой-либо 
    атрибут для какого-либо бассейна NaN, что также вызовет NaN's во время обучения модели.
    
    Параметры
    ----------
    df : pd.DataFrame
        DataFrame с атрибутами бассейнов в качестве столбцов.

    Вызывает
    ------
    RuntimeError
        Если один или несколько атрибутов имеют стандартное отклонение, равное нулю, или какой-либо атрибут для какого-либо бассейна является NaN.
    """
    # Проверка на NaN в стандартном отклонении атрибутов.
    attributes = []
    # ⚡ Bolt: Cache df.std() to avoid redundant expensive computations
    df_std = df.std()
    if any(df_std == 0.0) or any(df_std.isnull()):
        for k, v in df_std.items():
            if (v == 0) or (np.isnan(v)):
                attributes.append(k)
    if attributes:
        msg = [
            "Следующие атрибуты имеют стандартное отклонение, равное нулю или NaN, что приводит к NaN's ",
            "при нормализации признаков. Удалите атрибуты из списка признаков атрибутов ",
            "и перезапустите запуск. \n", f"Атрибуты: {attributes}"
        ]
        raise RuntimeError("".join(msg))

    # Проверка на NaN в любом атрибуте любого бассейна
    nan_df = df[df.isnull().any(axis=1)]
    if len(nan_df) > 0:
        failure_cases = defaultdict(list)
        for basin, row in nan_df.iterrows():
            for feature, value in row.items():
                if np.isnan(value):
                    failure_cases[basin].append(feature)
        # Создание подробного сообщения об ошибке
        msg = ["Следующие бассейны/атрибуты содержат NaN, которые не могут быть использованы в качестве входных данных:"]
        for basin, features in failure_cases.items():
            msg.append(f"{basin}: {features}")
        raise RuntimeError("\n".join(msg))


def sort_frequencies(frequencies: List[str]) -> List[str]:
    """Сортировка переданных частот от низких к высоким.

    Используйте `pandas frequency strings
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_
    для определения частот. Примечание: Строки должны включать значения, например, '1D' вместо 'D'.

    Параметры
    ----------
    frequencies : List[str]
        Список идентификаторов частот pandas для сортировки.

    Возвращает
    -------
    List[str]
        Отсортированный список идентификаторов частот pandas.

    Вызывает
    ------
    ValueError
        Если пара частот в `frequencies` не может быть сравнена через `compare_frequencies`.
    """
    return sorted(frequencies, key=functools.cmp_to_key(compare_frequencies))


def infer_frequency(index: Union[pd.DatetimeIndex, np.ndarray]) -> str:
    """Определение частоты индекса pandas DataFrame/Series или xarray DataArray.

    Параметры
    ----------
    index : Union[pd.DatetimeIndex, np.ndarray]
        DatetimeIndex DataFrame/Series или массив значений datetime.

    Возвращает
    -------
    str
        Частота индекса в виде `pandas frequency string
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

    Вызывает
    ------
    ValueError
        Если частота не может быть выведена из индекса или равна нулю.
    """
    native_frequency = pd.infer_freq(index)
    if native_frequency is None:
        raise ValueError(f'Не удается вывести допустимую частоту из набора данных: {native_frequency}.')
    if native_frequency[0] not in '0123456789':  # добавляем значение к единице, чтобы to_timedelta работала
        native_frequency = f'1{native_frequency}'

    # pd.Timedelta не понимает еженедельные (W) частоты, поэтому мы преобразуем их в эквивалентное кратное 7D.
    weekly_freq = re.match(r'(\d+)W(-(MON|TUE|WED|THU|FRI|SAT|SUN))?$', native_frequency)
    if weekly_freq is not None:
        n = int(weekly_freq[1]) * 7
        native_frequency = f'{n}D'

    # Проверяем, что частота соответствует положительному временному интервалу. Сначала мы добавляем один смещение к базовой дате,
    # чтобы убедиться, что она выровнена с частотой. В противном случае добавление смещения, например, 0Y, приведет к округлению до
    # ближайшего конца года, и мы ошибочно пропустим нулевую частоту.
    base_datetime = pd.to_datetime('2001-01-01 00:00:00') + to_offset(native_frequency)
    if base_datetime >= base_datetime + to_offset(native_frequency):
        raise ValueError('Выведенная частота набора данных равна нулю или отрицательна.')
    return native_frequency


def infer_datetime_coord(xr: Union[DataArray, Dataset]) -> str:
    """Проверяет наличие координаты с 'date' в названии и возвращает имя.
    
    Параметры
    ----------
    xr : Union[DataArray, Dataset]
        Массив для определения имени координаты.
        
    Возвращает
    -------
    str
        Имя координаты datetime.
        
    Вызывает
    ------
    RuntimeError
        Если не найдено ни одной координаты с 'date' в названии или найдено несколько.
    """
    candidates = [c for c in list(xr.coords) if "date" in c]
    if len(candidates) > 1:
        raise RuntimeError("Найдено несколько координат с 'date' в названии.")
    if not candidates:
        raise RuntimeError("Не найдено ни одной координаты с 'date' в названии")

    return candidates[0]


def compare_frequencies(freq_one: str, freq_two: str) -> int:
    """Сравнение двух частот.

    Примечание: Сравнивать можно только частоты, которые работают с `get_frequency_factor`.

    Параметры
    ----------
    freq_one : str
        Первая частота.
    freq_two : str
        Вторая частота.

    Возвращает
    -------
    int
        -1 если `freq_one` ниже `freq_two`, +1 если она выше, 0 если они равны.

    Вызывает
    ------
    ValueError
        Если две частоты не могут быть сравнены через `get_frequency_factor`.
    """
    freq_factor = get_frequency_factor(freq_one, freq_two)
    if freq_factor < 1:
        return 1
    if freq_factor > 1:
        return -1
    return 0


def get_frequency_factor(freq_one: str, freq_two: str) -> float:
    """Определение относительного коэффициента между двумя частотами.

    Параметры
    ----------
    freq_one : str
        Строковое представление первой частоты.
    freq_two : str
        Строковое представление второй частоты.

    Возвращает
    -------
    float
        Отношение `freq_one` к `freq_two`.

    Вызывает
    ------
    ValueError
        Если коэффициент частоты не может быть определен. Это может произойти, если частоты не представляют фиксированный
        временной интервал и не могут быть напрямую сравнены (например, потому что они имеют один и тот же юнит)
        Например, месяц не представляет фиксированный временной интервал. Таким образом, 1D и 1M не сравнимы. Однако, 1M и 2M
        сравнимы, так как имеют один и тот же юнит.
    """
    if freq_one == freq_two:
        return 1

    offset_one = to_offset(freq_one)
    offset_two = to_offset(freq_two)
    if offset_one.n < 0 or offset_two.n < 0:
        # Можно было бы реализовать, но нам никогда не понадобятся отрицательные частоты, поэтому кажется разумным
        # завершить работу с ошибкой, а не открывать себя для потенциальных неожиданных пограничных случаев.
        raise NotImplementedError('Невозможно сравнить отрицательные частоты.')
    # избегаем ошибок деления на ноль
    if offset_one.n == offset_two.n == 0:
        return 1
    if offset_two.n == 0:
        return np.inf
    if offset_one.name == offset_two.name:
        return offset_one.n / offset_two.n

    # некоторые простые жестко закодированные случаи
    factor = None
    regex_month_or_day = '-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|MON|TUE|WED|THU|FRI|SAT|SUN)$'
    for i, (one, two) in enumerate([(offset_one, offset_two), (offset_two, offset_one)]):
        # якорь смещения не имеет значения для отношения между частотами, поэтому мы удаляем его из строки
        name_one = re.sub(regex_month_or_day, '', one.name)
        name_two = re.sub(regex_month_or_day, '', two.name)
        if (name_one in ['A', _YE_FREQ] and name_two == _ME_FREQ) or (name_one in ['AS', 'YS'] and name_two == 'MS'):
            factor = 12 * one.n / two.n
        if (name_one in ['A', _YE_FREQ] and name_two == _QE_FREQ) or (name_one in ['AS', 'YS'] and name_two == 'QS'):
            factor = 4 * one.n / two.n
        if (name_one == _QE_FREQ and name_two == _ME_FREQ) or (name_one == 'QS' and name_two == 'MS'):
            factor = 3 * one.n / two.n
        if name_one == 'W' and name_two == 'D':
            factor = 7 * one.n / two.n

        if factor is not None:
            if i == 1:
                return 1 / factor  # `one` был `offset_two`, `two` был `offset_one`
            return factor

    # Если все остальные проверки не совпали, мы пытаемся преобразовать частоты в timedeltas. Однако, мы сначала должны избежать
    # двух случаев: (1) pd.to_timedelta в настоящее время интерпретирует 'M' как минуты, в то время как в to_offset он означает месяцы.
    # (2) Использование 'M', 'Y' и 'y' в pd.to_timedelta устарело и не будет работать в будущем, поэтому мы не разрешаем это.
    if any(
            re.sub(regex_month_or_day, '', offset.name) in ['M', 'Y', 'A', 'y', 'ME', 'YE']
            for offset in [offset_one, offset_two]):
        raise ValueError(f'Частоты {freq_one} и/or {freq_two} не сравнимы.')
    try:
        factor = pd.to_timedelta(freq_one) / pd.to_timedelta(freq_two)
    except ValueError as err:
        raise ValueError(f'Частоты {freq_one} и/or {freq_two} не сравнимы.') from err
    return factor
