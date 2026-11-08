import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning, njit, prange
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralprognoz.datautils import utils
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.errors import NoTrainDataError, NoEvaluationDataError
from neuralprognoz.utils import samplingutils

LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Базовый класс набора данных для загрузки и предварительной обработки данных."""

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' должен быть одним из 'train', 'validation' или 'test' ")
        else:
            self.period = period

        if period in ["validation", "test"]:
            if not scaler:
                raise ValueError("В период проверки или тестирования необходимо пройти проверку словаря масштабатора.")
            if cfg.use_basin_id_encoding and not id_to_int:
                raise ValueError("Для встраивания идентификатора бассейна словарь id_to_int должен быть передан любым способом, кроме train")

        if self.cfg.timestep_counter:
            if not self.cfg.forecast_inputs_flattened:
                raise ValueError('Счетчик временных шагов работает только для прогнозных данных.')
            if cfg.forecast_overlap:
                overlap_zeros = torch.zeros((cfg.forecast_overlap, 1))
                forecast_counter = torch.Tensor(range(1, cfg.forecast_seq_length - cfg.forecast_overlap + 1)).unsqueeze(-1)
                self.forecast_counter = torch.concatenate([overlap_zeros, forecast_counter], dim=0)
                self.hindcast_counter = torch.zeros((cfg.seq_length - cfg.forecast_seq_length + cfg.forecast_overlap, 1))
            else:
                self.forecast_counter = torch.Tensor(range(1, cfg.forecast_seq_length + 1)).unsqueeze(-1)
                self.hindcast_counter = torch.zeros((cfg.seq_length - cfg.forecast_seq_length, 1))

        if basin is None:
            self.basins = utils.load_basin_file(getattr(cfg, f"{period}_basin_file"))
        else:
            self.basins = [basin]
        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler
        
        if is_train and not scaler:
            self._compute_scaler = True
        else:
            self._compute_scaler = False

        self.frequencies = []
        self.seq_len = None
        self._predict_last_n = None
        self._initialize_frequency_configuration()

        self._disable_pbar = cfg.verbose == 0 or not self.is_train

        self._x_d = {}
        self._x_s = {}
        self._attributes = {}
        self._y = {}
        self._per_basin_target_stds = {}
        self._dates = {}
        self.start_and_end_dates = {}
        self.num_samples = 0
        self.period_starts = {}

        self._get_start_and_end_dates()

        if (not additional_features) and cfg.additional_feature_files:
            self._load_additional_features()

        if cfg.use_basin_id_encoding:
            if self.is_train:
                self._create_id_to_int()

        self._load_data()

        if self.is_train:
            self._dump_scaler()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int) -> dict:
        basin, indices = self.lookup_table[item]
        sample = {}
        for freq, seq_len, idx in zip(self.frequencies, self.seq_len, indices):
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            hindcast_start_idx = idx + 1 - seq_len
            global_end_idx = idx + 1
            if self.cfg.forecast_seq_length:
                hindcast_end_idx = idx + 1 - self.cfg.forecast_seq_length
                forecast_start_idx = idx + 1 - self.cfg.forecast_seq_length
                if self.cfg.forecast_overlap and self.cfg.forecast_overlap > 0:
                    hindcast_end_idx += self.cfg.forecast_overlap
            else:
                hindcast_end_idx = None
                forecast_start_idx = None
            x_d_key = f'x_d{freq_suffix}'
            sample[x_d_key] = {}
            sample[f'{x_d_key}_hindcast'] = {}
            sample[f'{x_d_key}_forecast'] = {}
            for k, v in self._x_d[basin][freq].items():
                if k in self.cfg.hindcast_inputs_flattened:
                    sample[f'{x_d_key}_hindcast'][k] = v[hindcast_start_idx:hindcast_end_idx]
                if k in self.cfg.forecast_inputs_flattened:
                    sample[f'{x_d_key}_forecast'][k] = v[forecast_start_idx:global_end_idx]
                if not self.cfg.hindcast_inputs_flattened:
                    sample[x_d_key][k] = v[hindcast_start_idx:global_end_idx]

            if self.is_train and (self.cfg.nan_step_probability or self.cfg.nan_sequence_probability):
                if self.cfg.hindcast_inputs_flattened:
                    sample[f'{x_d_key}_hindcast'] = self._add_nan_streaks(sample[f'{x_d_key}_hindcast'], groups=self.cfg.hindcast_inputs)
                    sample[f'{x_d_key}_forecast'] = self._add_nan_streaks(sample[f'{x_d_key}_forecast'], groups=self.cfg.forecast_inputs)
                else:
                    sample[x_d_key] = self._add_nan_streaks(sample[x_d_key], groups=self.cfg.dynamic_inputs)
            
            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

            if self.cfg.timestep_counter:
                sample[f'x_d{freq_suffix}']['hindcast_counter'] = self.hindcast_counter
                sample[f'x_d{freq_suffix}']['forecast_counter'] = self.forecast_counter

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(torch.tensor(self.id_to_int[basin]), num_classes=len(self.id_to_int)).to(torch.float32)

        return sample
    
    # ВОССТАНОВЛЕННЫЙ МЕТОД
    def _initialize_frequency_configuration(self):
        """Проверяет и извлекает значения конфигурации для 'use_ frequency', 'seq_length' и 'predict_last_n'"""
        self.frequencies = self.cfg.use_frequencies
        self.seq_len = self.cfg.seq_length
        self._predict_last_n = self.cfg.predict_last_n
        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length и predict_last_n должны быть целыми числами, если use_frequencies не указано.')
            self.seq_len = [self.seq_len]
            self._predict_last_n = [self._predict_last_n]
        else:
            if not isinstance(self.seq_len, dict) \
                    or not isinstance(self._predict_last_n, dict) \
                    or any([freq not in self.seq_len for freq in self.frequencies]) \
                    or any([freq not in self._predict_last_n for freq in self.frequencies]):
                raise ValueError('seq_length и predict_last_n должны быть словарями с одним ключом на частоту')
            self.seq_len = [self.seq_len[freq] for freq in self.frequencies]
            self._predict_last_n = [self._predict_last_n[freq] for freq in self.frequencies]

    def _add_nan_streaks(self, x_d: dict, groups: list) -> dict:
        if not groups or not isinstance(groups[0], list):
            raise ValueError('Для полос выпадения данных dynamic_inputs должен быть списком списков.')
        seq_length = x_d[groups[0][0]].shape[0]
        drop_masks = np.zeros((len(groups), seq_length, 1), dtype=bool)
        drop_sequences = np.random.choice([True, False], p=[self.cfg.nan_sequence_probability, 1-self.cfg.nan_sequence_probability], size=len(groups))
        if drop_sequences.all():
            drop_sequences[np.random.choice(len(groups))] = False
        for i in range(len(groups)):
            drop_steps = np.random.choice([True, False], p=[self.cfg.nan_step_probability, 1-self.cfg.nan_step_probability], size=(seq_length, 1))
            drop_masks[i] = drop_sequences[i] | drop_steps
        drop_masks = torch.from_numpy(drop_masks)
        for i, group in enumerate(groups):
            for feature in group:
                x_d[feature] = torch.where(drop_masks[i], torch.nan, x_d[feature])
        return x_d

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        raise NotImplementedError

    def _create_id_to_int(self):
        self.id_to_int = {str(b): i for i, b in enumerate(np.random.permutation(self.basins))}
        file_path = self.cfg.train_dir / "id_to_int.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(self.id_to_int, fp)

    def _dump_scaler(self):
        scaler = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, pd.Series):
                scaler[key] = value.to_dict()
            elif isinstance(value, xarray.Dataset):
                # Для Dataset сохраняется каждая переменная отдельно
                scaler[key] = {}
                for var_name in value.data_vars:
                    var_data = value[var_name]
                    scaler[key][var_name] = {
                        'dims': var_data.dims,
                        'data': var_data.values.tolist(),
                        'coords': {dim: var_data.coords[dim].values.tolist() for dim in var_data.dims}
                    }
            elif isinstance(value, xarray.DataArray):
                # Для DataArray сохраняется структура с координатами
                scaler[key] = {
                    'dims': value.dims,
                    'data': value.values.tolist(),
                    'coords': {dim: value.coords[dim].values.tolist() for dim in value.dims}
                }
            else:
                raise RuntimeError(f"Неизвестный тип данных для масштабирования: {key}. Поддерживаются объекты pd.Series и xarray.")
        
        file_path = self.cfg.train_dir / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler), fp)

    def _get_start_and_end_dates(self):
        """Получение начальной и конечной даты периода, обработка разделенных периодов."""
        
        # Определяем, какие даты использовать в зависимости от периода
        if self.is_train:
            start_dates_raw = self.cfg.train_start_date
            end_dates_raw = self.cfg.train_end_date
        else:
            if self.period == "validation":
                start_dates_raw = self.cfg.validation_start_date
                end_dates_raw = self.cfg.validation_end_date
            elif self.period == "test":
                start_dates_raw = self.cfg.test_start_date
                end_dates_raw = self.cfg.test_end_date
            else:
                # По умолчанию используем validation
                start_dates_raw = self.cfg.validation_start_date
                end_dates_raw = self.cfg.validation_end_date
        
        # Обрабатываем даты в зависимости от формата
        if isinstance(start_dates_raw, list) and isinstance(end_dates_raw, list):
            # Разделенные периоды (как у вас в конфиге)
            if len(start_dates_raw) != len(end_dates_raw):
                raise ValueError(f"Количество начальных дат ({len(start_dates_raw)}) "
                               f"не совпадает с количеством конечных дат ({len(end_dates_raw)})")
            
            # Преобразуем в datetime если нужно
            start_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d 
                          for d in start_dates_raw]
            end_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d 
                        for d in end_dates_raw]
            
            if len(start_dates) > 1:
                # Множественные периоды
                self.split_periods = list(zip(start_dates, end_dates))
                LOGGER.info(f"Используются разделенные периоды: {len(self.split_periods)} сегментов")
            else:
                # Один период, но в формате списка
                self.split_periods = None
                
        else:
            # Простой период (одна дата начала и одна дата конца)
            start_dates = [pd.to_datetime(start_dates_raw) if not isinstance(start_dates_raw, pd.Timestamp) 
                          else start_dates_raw]
            end_dates = [pd.to_datetime(end_dates_raw) if not isinstance(end_dates_raw, pd.Timestamp) 
                        else end_dates_raw]
            self.split_periods = None
        
        # Сохраняем даты для каждого бассейна
        for basin in self.basins:
            self.start_and_end_dates[basin] = {
                "start_dates": start_dates,
                "end_dates": end_dates
            }
    def _load_additional_features(self):
        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                LOGGER.warning("SECURITY WARNING: Loading a pickle file can execute arbitrary code. "
                               "Ensure that the feature file comes from a trusted source.")
                self.additional_features.append(pickle.load(fp))

    def _duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, n_duplicates in self.cfg.duplicate_features.items():
            for n in range(1, n_duplicates + 1):
                df[f"{feature}_copy{n}"] = df[feature]
        return df
    
    def _add_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.cfg.target_variables:
            if var not in df.columns:
                df[var] = np.nan
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_autoregressive_inputs()
        for feature, shift in self.cfg.lagged_features.items():
            if isinstance(shift, list):
                for s in set(shift):
                    df[f"{feature}_shift{s}"] = df[feature].shift(periods=s, freq="infer")
            elif isinstance(shift, int):
                df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift, freq="infer")
            else:
                raise ValueError("Значение аргумента 'lagged_features' должно быть либо целым числом, либо списком целых чисел.")
        return df

    def _check_autoregressive_inputs(self):
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Авторегрессионные входные данные должны быть сдвинутой переменной в форме <variable>_shift<lag> ',
                                 f'где <lag> integer. Вместо этого есть: {input}.')
            if capture[1] not in self.cfg.lagged_features or int(capture[2]) not in self.cfg.lagged_features[capture[1]]:
                raise ValueError('Авторегрессионные входы должны быть в списке "lagged_inputs".')
        return

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []
            keep_cols = self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.mass_inputs + self.cfg.autoregressive_inputs
            if isinstance(self.cfg.dynamic_inputs, list):
                keep_cols += self.cfg.dynamic_inputs_flattened
            else:
                keep_cols += [i for inputs in self.cfg.dynamic_inputs.values() for i in inputs]
            keep_cols += self.cfg.dynamic_conceptual_inputs
            keep_cols = list(sorted(set(keep_cols)))
            if not self._disable_pbar:
                LOGGER.info("Загрузка данных бассейнов в xarray dataset.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):
                df = self._load_basin_data(basin)
                df = pd.concat([df, *[d[basin] for d in self.additional_features]], axis=1)
                if not self.is_train:
                    df = self._add_missing_targets(df)
                df = self._duplicate_features(df)
                df = self._add_lagged_features(df)
                try:
                    df = df[keep_cols]
                except KeyError:
                    not_available_columns = [x for x in keep_cols if x not in df.columns]
                    msg = [f"Следующие функции недоступны в данных: {not_available_columns}. ",
                           f"Доступны функции: {df.columns.tolist()}"]
                    raise KeyError("".join(msg))
                for holdout_variable, holdout_dict in self.cfg.random_holdout_from_dynamic_features.items():
                    df[holdout_variable] = samplingutils.bernoulli_subseries_sampler(
                        data=df[holdout_variable].values,
                        missing_fraction=holdout_dict['missing_fraction'],
                        mean_missing_length=holdout_dict['mean_missing_length'],
                    )
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]]
                native_frequency = utils.infer_frequency(df.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]
                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Невозможно сравнить предоставленные частоты с родной частотой. Убедитесь, что частоты не превышают родную частоту.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Частота выше, чем собственная частота данных {native_frequency}.')
                offsets = [(self.seq_len[i] - self._predict_last_n[i]) * to_offset(freq) for i, freq in enumerate(self.frequencies)]
                basin_data_list = []
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        misaligned = [freq for freq in self.frequencies if not to_offset(freq).is_on_offset(start_date)]
                        raise ValueError(f'Дата начала {start_date} не согласована с частотами {misaligned}.')
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_sub = df[warmup_start_date:end_date]
                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_sub = df_sub.reindex(pd.DatetimeIndex(full_range, name=df_sub.index.name))
                    df_sub.loc[df_sub.index < start_date, self.cfg.target_variables] = np.nan
                    basin_data_list.append(df_sub)
                if not basin_data_list:
                    continue
                df = pd.concat(basin_data_list, axis=0)
                df_non_duplicated = df[~df.index.duplicated(keep=False)]
                df_duplicated = df[df.index.duplicated(keep=False)]
                filtered_duplicates = []
                for _, grp in df_duplicated.groupby('date'):
                    mask = ~grp[self.cfg.target_variables].isna().any(axis=1)
                    if not mask.any():
                        filtered_duplicates.append(grp.head(1))
                    else:
                        filtered_duplicates.append(grp[mask].head(1))
                if filtered_duplicates:
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    df = df_non_duplicated
                df = df.sort_index(axis=0, ascending=True)
                df = df.reindex(pd.DatetimeIndex(data=pd.date_range(df.index[0], df.index[-1], freq=native_frequency), name=df.index.name))
                
                dtype_map = {col: np.float32 for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
                xr = xarray.Dataset.from_dataframe(df.astype(dtype_map))
                
                xr = xr.assign_coords({'basin': basin})
                data_list.append(xr)
            if not data_list:
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError
            xr = xarray.concat(data_list, dim="basin")
            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr)
        else:
            with self.cfg.train_data_file.open("rb") as fp:
                LOGGER.warning("SECURITY WARNING: Loading a pickle file can execute arbitrary code. "
                               "Ensure that the dataset file comes from a trusted source.")
                d = pickle.load(fp)
            xr = xarray.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr["date"].values)
                self.frequencies = [native_frequency]
        return xr

    def _save_xarray_dataset(self, xr: xarray.Dataset):
        file_path = self.cfg.train_dir / "train_data.p"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as fp:
            pickle.dump(xr.to_dict(), fp)

    def _calculate_per_basin_std(self, xr: xarray.Dataset):
        if not self._disable_pbar:
            LOGGER.info("Расчет целевых переменных стандартных отклонений для каждого бассейна")
        nan_basins = []
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):
            obs = xr.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                per_basin_target_stds = torch.tensor(np.expand_dims(np.nanstd(obs, axis=1), 0), dtype=torch.float32)
            else:
                nan_basins.append(basin)
                per_basin_target_stds = torch.full((1, obs.shape[0]), np.nan, dtype=torch.float32)
            self._per_basin_target_stds[basin] = per_basin_target_stds
        if len(nan_basins) > 0:
            LOGGER.warning(f"В следующих бассейнах не было достаточно допустимых целевых значений: {', '.join(nan_basins)}. Потеря NSE составит NaN.")

    def _create_lookup_table(self, xr: xarray.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Создать таблицу поиска и преобразовать в pytorch tensor")
        basins_without_samples = []
        basin_coordinates = xr["basin"].values.tolist()
        
        # Проверить типы данных для всех переменных
        if self.is_train:
            LOGGER.info("Проверка типов данных для всех переменных...")
            for var_name in xr.data_vars:
                var_dtype = xr[var_name].dtype
                if not np.issubdtype(var_dtype, np.number) or np.issubdtype(var_dtype, np.timedelta64):
                    LOGGER.warning(f"Переменная {var_name} имеет нечисловой тип: {var_dtype}")
        
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            x_d, x_s, y, dates = {}, {}, {}, {}
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]
            
            # Извлечение данных для конкретного бассейна и создание DataFrame с DatetimeIndex
            basin_data = xr.sel(basin=basin)
            df_native = basin_data.to_dataframe()
            
            # Если индекс MultiIndex, извлечь только DatetimeIndex
            if isinstance(df_native.index, pd.MultiIndex):
                # Проверка, какие уровни есть в MultiIndex
                index_levels = df_native.index.names
                LOGGER.debug(f"MultiIndex levels: {index_levels}")
                
                # Убрать все уровни кроме date, если они есть
                levels_to_drop = [level for level in index_levels if level is not None and level != 'date']
                if levels_to_drop:
                    df_native = df_native.reset_index(level=levels_to_drop, drop=True)
            
            # Убедиться, что индекс - это DatetimeIndex
            if not isinstance(df_native.index, pd.DatetimeIndex):
                # Если индекс не DatetimeIndex, попытка преобразовать его
                try:
                    df_native.index = pd.to_datetime(df_native.index)
                except Exception as e:
                    LOGGER.warning(f"Не удалось преобразовать индекс в DatetimeIndex: {e}")
                    # Создание нового DatetimeIndex на основе данных
                    if 'date' in df_native.columns:
                        df_native = df_native.set_index('date')
                    else:
                        # Создание последовательного DatetimeIndex
                        df_native.index = pd.date_range('1993-01-01', periods=len(df_native), freq='D')
            for freq in self.frequencies:
                if isinstance(self.cfg.dynamic_inputs, list):
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs_flattened
                else:
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs[freq]
                dynamic_cols += self.cfg.dynamic_conceptual_inputs
                
                # Фильтр только числовых колонок, исключая TimeDelta64
                numeric_cols = []
                for col in dynamic_cols:
                    if col in df_native.columns:
                        if (np.issubdtype(df_native[col].dtype, np.number) and 
                            not np.issubdtype(df_native[col].dtype, np.timedelta64)):
                            numeric_cols.append(col)
                        else:
                            LOGGER.warning(f"Пропуск нечисловой колонки {col} с типом {df_native[col].dtype}")
                
                df_resampled = df_native[numeric_cols + self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.autoregressive_inputs].resample(freq).mean()
                x_d[freq] = {col: df_resampled[[col]].values for col in numeric_cols}
                y[freq] = df_resampled[self.cfg.target_variables].values
                if self.cfg.evolving_attributes:
                    x_s[freq] = df_resampled[self.cfg.evolving_attributes].values
                dates[freq] = df_resampled.index.to_numpy()
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                if len(df_resampled) % frequency_factor != 0:
                    raise ValueError(f'Длина датафрейма в частоте {freq} составляет {len(df_resampled)} (включая разогрев), не кратное {frequency_factor}.')
                frequency_maps[freq] = np.arange(len(df_resampled) // frequency_factor) * frequency_factor + (frequency_factor - 1)
            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(xr.sel(basin=basin)["date"].values[0])
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
                if self.is_train:
                    # Проверка, что все значения в x_d[freq] числовые
                    x_d_validate = []
                    for freq in self.frequencies:
                        freq_values = []
                        for v in x_d[freq].values():
                            if np.issubdtype(v.dtype, np.number) and not np.issubdtype(v.dtype, np.timedelta64):
                                freq_values.append(v)
                            else:
                                LOGGER.warning(f"Пропуск нечисловых значений с типом {v.dtype} в частоте {freq}")
                        if freq_values:
                            x_d_validate.append(np.concatenate(freq_values, axis=-1))
                        else:
                            LOGGER.error(f"Нет числовых значений для частоты {freq}")
                            x_d_validate.append(np.array([]))
                else:
                    x_d_validate = None
                flag = _validate_samples(x_d=x_d_validate,
                                        x_s=[x_s[freq] for freq in self.frequencies] if self.is_train and x_s else None,
                                        y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                                        frequency_maps=[frequency_maps[freq] for freq in self.frequencies],
                                        seq_length=self.seq_len,
                                        predict_last_n=self._predict_last_n)
            if self.cfg.autoregressive_inputs:
                if len(self.frequencies) > 1:
                    raise ValueError('Авторегрессионные входные данные не поддерживаются для многочастотных наборов данных.')
                
                # Фильтр autoregressive_inputs, исключая нечисловые переменные
                numeric_autoregressive = []
                for col in self.cfg.autoregressive_inputs:
                    if col in df_resampled.columns:
                        if (np.issubdtype(df_resampled[col].dtype, np.number) and 
                            not np.issubdtype(df_resampled[col].dtype, np.timedelta64)):
                            numeric_autoregressive.append(col)
                        else:
                            LOGGER.warning(f"Пропуск нечисловой autoregressive колонки {col} с типом {df_resampled[col].dtype}")
                
                if numeric_autoregressive:
                    x_d[self.frequencies[0]].update({col: df_resampled[[col]].values for col in numeric_autoregressive})
            valid_samples = np.argwhere(flag == 1)
            for f in valid_samples:
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))
            if valid_samples.size > 0:
                self._x_d[basin] = {freq: {k: torch.from_numpy(v.astype(np.float32)) for k, v in _x_d.items()} for freq, _x_d in x_d.items()}
                self._y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                if x_s:
                    self._x_s[basin] = {freq: torch.from_numpy(_x_s.astype(np.float32)) for freq, _x_s in x_s.items()}
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)
        if basins_without_samples:
            LOGGER.info(f"В этих бассейнах нет достоверных образцов {self.period}: {basins_without_samples}")
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

    def _load_hydroatlas_attributes(self):
        df = utils.load_hydroatlas_attributes(self.cfg.data_dir, basins=self.basins)
        drop_cols = [c for c in df.columns if c not in self.cfg.hydroatlas_attributes]
        df = df.drop(drop_cols, axis=1)
        if self.is_train:
            utils.attributes_sanity_check(df=df)
        return df

    def _load_combined_attributes(self):
        dfs = []
        if self.cfg.static_attributes:
            df = self._load_attributes()
            missing_attrs = [attr for attr in self.cfg.static_attributes if attr not in df.columns]
            if len(missing_attrs) > 0:
                raise ValueError(f'Статические атрибуты {missing_attrs} отсутствуют.')
            df = df[self.cfg.static_attributes]
            if self._compute_scaler:
                utils.attributes_sanity_check(df=df)
            dfs.append(df)
        if self.cfg.hydroatlas_attributes:
            dfs.append(self._load_hydroatlas_attributes())
        if dfs:
            df = pd.concat(dfs, axis=1)
            combined_attributes = self.cfg.static_attributes + self.cfg.hydroatlas_attributes
            missing_columns = [attr for attr in combined_attributes if attr not in df.columns]
            if missing_columns:
                raise ValueError(f"Следующие атрибуты недоступны: {missing_columns}")
            df = df.sort_index(axis=1)
            if self._compute_scaler:
                self.scaler["attribute_means"] = df.mean()
                self.scaler["attribute_stds"] = df.std()
            if any([k.startswith("camels_attr") for k in self.scaler.keys()]):
                LOGGER.warning("Предупреждение об устаревании: старые файлы масштабирования скоро не будут поддерживаться.")
                df = (df - self.scaler['camels_attr_means']) / self.scaler["camels_attr_stds"]
            else:
                df = (df - self.scaler['attribute_means']) / self.scaler["attribute_stds"]
            for basin in self.basins:
                attributes = df.loc[df.index == basin].values.flatten()
                self._attributes[basin] = torch.from_numpy(attributes.astype(np.float32))

    def _load_data(self):
        self._load_combined_attributes()
        xr = self._load_or_create_xarray_dataset()
        if self.cfg.loss.lower() in ['nse', 'weightednse']:
            self._calculate_per_basin_std(xr)
        if self._compute_scaler:
            self._setup_normalization(xr)
        
        # Собираем список только числовых переменных для нормализации
        all_input_vars = self.cfg.target_variables + self.cfg.dynamic_inputs_flattened
        vars_to_normalize = []
        for v in sorted(list(set(all_input_vars))):
            if v in xr.data_vars:
                # Проверяем, что переменная числовая и не является TimeDelta64
                if (np.issubdtype(xr[v].dtype, np.number) and 
                    not np.issubdtype(xr[v].dtype, np.timedelta64)):
                    vars_to_normalize.append(v)
        
        if vars_to_normalize:
            # Создаем под-набор данных только с этими переменными
            xr_numeric = xr[vars_to_normalize]

            # Нормализуем только этот под-набор
            xr_normalized = (xr_numeric - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]
            
            # Обновляем исходный Dataset нормализованными значениями. Нечисловые переменные остаются нетронутыми.
            xr.update(xr_normalized)
        else:
            LOGGER.warning("Нет числовых переменных для нормализации в _load_data")
        
        self._create_lookup_table(xr)

    def _setup_normalization(self, xr: xarray.Dataset):
        """Get feature-wise center and scale values for feature normalization"""
        all_input_vars = self.cfg.target_variables + self.cfg.dynamic_inputs_flattened
        
        # Фильтруем список, оставляя только существующие в датасете и числовые переменные
        # Исключаем TimeDelta64 и другие нечисловые типы
        vars_to_normalize = []
        for v in sorted(list(set(all_input_vars))):
            if v in xr.data_vars:
                # Проверяем, что переменная числовая и не является TimeDelta64
                if (np.issubdtype(xr[v].dtype, np.number) and 
                    not np.issubdtype(xr[v].dtype, np.timedelta64)):
                    vars_to_normalize.append(v)
        
        missing_vars = [v for v in all_input_vars if v not in xr.data_vars]
        if missing_vars:
            LOGGER.warning(f"Переменные из конфига не найдены в датасете и будут проигнорированы: {missing_vars}")
        
        if not vars_to_normalize:
            LOGGER.warning("Нет числовых переменных для нормализации")
            return
            
        LOGGER.info(f"Переменные для нормализации: {vars_to_normalize}")
        xr_subset = xr[vars_to_normalize]
        
        # Дополнительная проверка на случай, если xr_subset пустой
        if len(xr_subset.data_vars) == 0:
            LOGGER.warning("xr_subset пустой, пропускаем нормализацию")
            return
            
        # Создаем scaler для каждой переменной отдельно, сохраняя измерение variable
        feature_center_list = []
        feature_scale_list = []
        
        for var in vars_to_normalize:
            var_data = xr_subset[var]
            center = var_data.mean(dim="date", skipna=True)
            scale = var_data.std(dim="date", skipna=True)
            feature_center_list.append(center)
            feature_scale_list.append(scale)
        
        # Объединяем в DataArray с измерением variable
        self.scaler["xarray_feature_center"] = xarray.concat(feature_center_list, dim="variable").assign_coords(variable=vars_to_normalize)
        self.scaler["xarray_feature_scale"] = xarray.concat(feature_scale_list, dim="variable").assign_coords(variable=vars_to_normalize)
        
        for feature, feature_specs in self.cfg.custom_normalization.items():
            if feature in vars_to_normalize:
                for key, val in feature_specs.items():
                    if key == "centering":
                        if (val is None) or (val.lower() == "none"):
                            self.scaler["xarray_feature_center"].loc[dict(variable=feature)] = np.float32(0.0)
                        elif val.lower() == "median":
                            self.scaler["xarray_feature_center"].loc[dict(variable=feature)] = xr[feature].median(skipna=True)
                        elif val.lower() == "min":
                            self.scaler["xarray_feature_center"].loc[dict(variable=feature)] = xr[feature].min(skipna=True)
                        elif val.lower() == "mean":
                            pass
                        else:
                            raise ValueError(f"Неизвестный метод центрирования {val}")
                    elif key == "scaling":
                        if (val is None) or (val.lower() == "none"):
                            self.scaler["xarray_feature_scale"].loc[dict(variable=feature)] = np.float32(1.0)
                        elif val == "minmax":
                            scale_val = xr[feature].max(skipna=True) - xr[feature].min(skipna=True)
                            self.scaler["xarray_feature_scale"].loc[dict(variable=feature)] = scale_val
                        elif val == "std":
                            pass
                        else:
                            raise ValueError(f"Неизвестный метод скейлинга {val}")
                    else:
                        raise ValueError("Неизвестный dict key. Использовать 'centering' и/или 'scaling'.")

    def get_period_start(self, basin: str) -> pd.Timestamp:
        return self.period_starts[basin]

    @staticmethod
    def collate_fn(samples: List[Dict]) -> Dict:
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith('date'):
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            elif feature.startswith('x_d'):
                batch[feature] = {k: torch.stack([sample[feature][k] for sample in samples], dim=0) for k in samples[0][feature]}
            else:
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch

@njit()
def _validate_samples(x_d: List[np.ndarray], x_s: List[np.ndarray], y: List[np.ndarray], seq_length: List[int],
                     predict_last_n: List[int], frequency_maps: List[np.ndarray]) -> np.ndarray:
    n_samples = len(frequency_maps[0])
    flag = np.ones(n_samples)
    for i in range(len(frequency_maps)):
        for j in prange(n_samples):
            last_sample_of_freq = frequency_maps[i][j]
            if last_sample_of_freq < seq_length[i] - 1:
                flag[j] = 0
                continue
            if x_d is not None:
                _x_d = x_d[i][last_sample_of_freq - seq_length[i] + 1:last_sample_of_freq + 1]
                if np.any(np.isnan(_x_d)):
                    flag[j] = 0
                    continue
            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue
            if x_s is not None:
                _x_s = x_s[i][last_sample_of_freq]
                if np.any(np.isnan(_x_s)):
                    flag[j] = 0
    return flag
