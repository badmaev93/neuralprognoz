from functools import reduce
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray

from neuralprognoz.datasetzoo.basedataset import BaseDataset
from neuralprognoz.utils.config import Config

class GenericDataset(BaseDataset):
    """Класс набора данных для универсального набора данных, который читает данные для любого региона на основе общих соглашений о структуре файлов.

    Для использования этого набора данных data_dir должен содержать папку 'time_series' и (если используются статические атрибуты) папку
    'attributes'. Папка 'time_series' содержит один netcdf-файл (.nc или .nc4) на бассейн, названный '<basin_id>.nc/nc4'.
    Netcdf-файл должен иметь одну координату с именем `date`, содержащую индекс datetime. Папка 'attributes' 
    содержит один или несколько файлов с разделителями-запятыми (.csv) со статическими атрибутами, индексированными по идентификатору бассейна. Файлы атрибутов могут 
    быть разделены на группы бассейнов или группы признаков (но не оба, см. `genericdataset.load_attributes` для
    более подробной информации).

    Примечание: Некорректные значения должны быть помечены как NaN (например, используя NumPy's np.nan) в netCDF-файлах, а не что-то вроде
    -999 для некорректных измерений расхода, что часто встречается в гидрологических наборах данных. Если отсутствующие значения не 
    помечены как NaN, GenericDataset не сможет идентифицировать эти значения как отсутствующие точки данных.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    is_train : bool
        Определяет, используется ли набор данных для обучения или оценки. Если True (обучение), средние/стандартные отклонения
        для каждой характеристики вычисляются и сохраняются в директории запуска. Если используется one-hot кодирование,
        создаётся отображение для one-hot кодирования и также сохраняется на диск. Если False, ожидается входной параметр `scaler`
        и аналогично `id_to_int`, если используется one-hot кодирование.
    period : {'train', 'validation', 'test'}
        Определяет период, для которого будут загружены данные
    basin : str, optional
        Если указано, данные будут загружены только для этого бассейна. В противном случае бассейны читаются из соответствующего
        файла бассейнов, соответствующего `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        Список словарей, отображающих идентификатор бассейна в pandas DataFrame. Этот DataFrame будет добавлен к данным,
        загруженным из набора данных, и все столбцы будут доступны как 'dynamic_inputs', 'evolving_attributes' и
        'target_variables'
    id_to_int : Dict[str, int], optional
        Если в конфиге аргумент 'use_basin_id_encoding' равен True и период — 'validation' или 'test', этот параметр обязателен.
        Это словарь, отображающий идентификатор бассейна в целое число (one-hot кодирование).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        Если период — 'validation' или 'test', этот параметр обязателен. Содержит центрирование и масштабирование
        для каждой характеристики и сохраняется в директории запуска во время обучения (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(GenericDataset, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Загрузить входные и выходные данные."""
        df = load_timeseries(data_dir=self.cfg.data_dir, basin=basin)

        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Загрузить статические атрибуты водосбора."""
        return load_attributes(self.cfg.data_dir, basins=self.basins)

def load_attributes(data_dir: Path, basins: List[str] = None) -> pd.DataFrame:
    """Загрузить статические атрибуты.

    Параметры
    ----------
    data_dir : Path
        Путь к директории данных. Эта папка должна содержать папку 'attributes' с одним или несколькими csv-файлами.
    basins : List[str], optional
        Если указано, возвращать только атрибуты для бассейнов, указанных в этом списке. В противном случае возвращаются атрибуты
        всех бассейнов.

    Возвращает
    -------
    pandas.DataFrame
        DataFrame с индексом по бассейнам, содержащий атрибуты в качестве столбцов. Если папка атрибутов содержит несколько
        файлов, они будут объединены следующим образом:
        
        (a) если пересечение бассейнов непустое, атрибуты файлов объединяются для пересечения
            бассейнов. Пересечение атрибутов должно быть пустым в этом случае.
        (b) если пересечение бассейнов пустое, но пересечение атрибутов непустое, бассейны файлов
            объединяются для пересечения атрибутов.
            
        Во всех остальных случаях выбрасывается ValueError.

    Вызывает
    ------
    FileNotFoundError
        Если папка атрибутов не найдена или не содержит csv-файлов.
    ValueError
        Если файл атрибутов содержит дублирующиеся имена бассейнов или признаков, найдены несколько файлов без
        перекрытия, или отсутствуют атрибуты для бассейна, указанного в `basins`.
    """
    attributes_path = data_dir / 'attributes'
    if not attributes_path.exists():
        raise FileNotFoundError(f"Папка атрибутов не найдена по пути {attributes_path}")

    files = list(attributes_path.glob('*.csv'))
    if not files:
        raise FileNotFoundError('Файлы атрибутов не найдены')

    # Читаем атрибуты в один большой dataframe. Сортируем по обеим осям, чтобы можно было проверить идентичные оси.
    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={0: str})  # убедиться, что мы читаем идентификатор бассейна как str
        df = df.set_index(df.columns[0]).sort_index(axis=0).sort_index(axis=1)
        if df.index.has_duplicates or df.columns.has_duplicates:
            raise ValueError(f'Файл атрибутов {f} содержит дублирующиеся идентификаторы бассейнов или признаки.')
        dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        if len(reduce(lambda idx, other_idx: idx.intersection(other_idx), (df.index for df in dfs))) > 0:
            # пересечение бассейнов непустое -> объединяем атрибуты, оставляем пересечение бассейнов
            if np.any(np.unique(np.concatenate([df.columns for df in dfs]), return_counts=True)[1] > 1):
                raise ValueError('Если dataframe атрибутов относятся к одним и тем же бассейнам, ни одно имя атрибута не может встречаться '
                                 'несколько раз в разных файлах атрибутов.')
            concat_axis = 1
        elif len(reduce(lambda cols, other_cols: cols.intersection(other_cols), (df.columns for df in dfs))) > 0:
            # пересечение атрибутов непустое -> объединяем бассейны, оставляем пересечение атрибутов
            # не нужно проверять дубликаты бассейнов, поскольку тогда у нас было бы непустое пересечение бассейнов.
            concat_axis = 0
        else:
            raise ValueError('Файлы атрибутов должны перекрываться либо по индексу, либо по столбцам.')

        df = pd.concat(dfs, axis=concat_axis, join='inner')

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Некоторые бассейны не имеют статических атрибутов.')
        df = df.loc[basins]

    return df

def load_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    # ⚡ Bolt: Lazily evaluate glob using `next()` to stop searching once the file is found.
    # This avoids the O(N) overhead of casting the entire recursive search tree to a list
    # while maintaining the structural flexibility of searching across subdirectories.
    netcdf_file = next(data_dir.glob(f"**/{basin}.nc"), None)
    if netcdf_file is None:
        netcdf_file = next(data_dir.glob(f"**/{basin}.nc4"), None)
        if netcdf_file is None:
            raise FileNotFoundError(f"No NetCDF file found for basin {basin} in {data_dir}")

    xr_ds = None
    try:
        with xarray.open_dataset(netcdf_file, decode_timedelta=False, engine="h5netcdf") as ds_temp:
            xr_ds = ds_temp.load()
    except Exception as e1:
        try:
            with xarray.open_dataset(netcdf_file, decode_timedelta=False) as ds_temp:
                xr_ds = ds_temp.load()
        except Exception as e2:
            raise RuntimeError(f"Failed to load {basin}.nc. Errors: {e1}, {e2}")
    if xr_ds is None:
        raise RuntimeError(f"Failed to load {basin}.nc (Unknown error)")
    df = xr_ds.to_dataframe()
    if isinstance(df.index, pd.MultiIndex):
        if "date" in df.index.names:
            df = df.reset_index(level=[n for n in df.index.names if n != "date"], drop=True)
        elif "time" in df.index.names:
            df = df.reset_index(level=[n for n in df.index.names if n != "time"], drop=True)
    return df
