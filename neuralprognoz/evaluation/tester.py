import logging
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralprognoz.datasetzoo import get_dataset
from neuralprognoz.datasetzoo.basedataset import BaseDataset
from neuralprognoz.datautils.utils import get_frequency_factor, load_basin_file, load_scaler, sort_frequencies
from neuralprognoz.evaluation import plots
from neuralprognoz.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralprognoz.evaluation.utils import load_basin_id_encoding, metrics_to_dataframe
from neuralprognoz.modelzoo import get_model
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.training import get_loss_obj, get_regularization_obj
from neuralprognoz.training.logger import Logger
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.errors import AllNaNError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)


class BaseTester(object):
    """Базовый класс для запуска инференса модели.

    Используйте подклассы этого класса для оценки обученной модели на ее обучающем, тестовом или валидационном периоде.
    Для регрессионных задач используется `RegressionTester`; для предсказания неопределенности — `UncertaintyTester`.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    run_dir : Path
        Путь к директории запуска.
    period : {'train', 'validation', 'test'}, optional
        Период для оценки, по умолчанию 'test'.
    init_model : bool, optional
        Если True, веса модели будут инициализированы чекпоинтом из последнего доступного эпохи в `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model
        if period in ["train", "validation", "test"]:
            self.period = period
        else:
            raise ValueError(f'Неверный период {period}. Должен быть одним из ["train", "validation", "test"]')

        # определение устройства
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        self._disable_pbar = cfg.verbose == 0

        # предварительная инициализация переменных, определенных в методах класса
        self.basins = None
        self.scaler = None
        self.id_to_int = {}
        self.additional_features = []

        # заполнитель для хранения кэшированных данных проверки
        self.cached_datasets = {}

        # объект потерь для вычисления потерь данных оценки
        self.loss_obj = get_loss_obj(cfg)
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

        self._load_run_data()

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"На этой машине нет GPU #{gpu_id}")
                else:
                    self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS устройство недоступно.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _load_run_data(self):
        """Загрузка данных запуска из директории запуска"""

        # получение списка бассейнов
        self.basins = load_basin_file(getattr(self.cfg, f"{self.period}_basin_file"))

        # загрузка масштабировщика признаков
        self.scaler = load_scaler(self.run_dir)

        # проверка старых файлов масштабировщика, где параметры центра/масштаба имели старые имена
        if "xarray_means" in self.scaler.keys():
            self.scaler["xarray_feature_center"] = self.scaler.pop("xarray_means")
        if "xarray_stds" in self.scaler.keys():
            self.scaler["xarray_feature_scale"] = self.scaler.pop("xarray_stds")

        # загрузка словаря basin_id в целое число для one-hot-encoding
        if self.cfg.use_basin_id_encoding:
            self.id_to_int = load_basin_id_encoding(self.run_dir)

        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _get_weight_file(self, epoch: int):
        """Получение пути к файлу весов"""
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob('model_epoch*.pt')))[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    def _load_weights(self, epoch: int = None):
        """Загрузка весов определенной (или последней) эпохи в модель."""
        weight_file = self._get_weight_file(epoch)

        LOGGER.info(f"Используются веса модели из {weight_file}")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device, weights_only=True))

    def _get_dataset(self, basin: str) -> BaseDataset:
        """Получение датасета для одного бассейна."""
        ds = get_dataset(cfg=self.cfg,
                         is_train=False,
                         period=self.period,
                         basin=basin,
                         additional_features=self.additional_features,
                         id_to_int=self.id_to_int,
                         scaler=self.scaler)
        return ds

    def evaluate(self,
                 epoch: int = None,
                 save_results: bool = True,
                 save_all_output: bool = False,
                 metrics: Union[list, dict] = [],
                 model: torch.nn.Module = None,
                 experiment_logger: Logger = None) -> dict:
        """Оценка модели.
        
        Параметры
        ----------
        epoch : int, optional
            Определяет конкретную эпоху для оценки. По умолчанию используются веса последней эпохи.
        save_results : bool, optional
            Если True, сохраняет результаты оценки в директории запуска. По умолчанию True.
        save_all_output : bool, optional
            Если True, сохраняет весь вывод модели в директории запуска. По умолчанию False.
        metrics : Union[list, dict], optional
            Список метрик для вычисления во время оценки. Также может быть словарем, который указывает метрики для каждой цели.
        model : torch.nn.Module, optional
            Если передана модель, она используется для валидации.
        experiment_logger : Logger, optional
            Логгер может быть передан во время обучения для логирования метрик.

        Возвращает
        -------
        dict
            Словарь, содержащий один xarray на бассейн с результатами оценки.
        """
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError("Модель не была инициализирована для оценки")

        # во время валидации, в зависимости от настроек, оцениваем только на случайной подвыборке бассейнов
        basins = self.basins
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[:self.cfg.validate_n_random_basins]

        # принудительный перевод модели в режим обучения при выполнении mc-dropout оценки
        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        results = defaultdict(dict)
        all_output = {basin: None for basin in basins}

        pbar = tqdm(basins, file=sys.stdout, disable=self._disable_pbar)
        pbar.set_description('# Validation' if self.period == "validation" else "# Evaluation")

        for basin in pbar:

            if self.cfg.cache_validation_data and basin in self.cached_datasets.keys():
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoEvaluationDataError as error:
                    # пропуск бассейна
                    continue
                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=0, pin_memory=self.device.type != "cpu", collate_fn=ds.collate_fn)

            y_hat, y, dates, all_losses, all_output[basin] = self._evaluate(model, loader, ds.frequencies,
                                                                            save_all_output)

            # логирование потерь этого бассейна плюс количества образцов в логгере для последующего вычисления агрегатов эпохи
            if experiment_logger is not None:
                experiment_logger.log_step(**{k: (v, len(loader)) for k, v in all_losses.items()})

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length
            # если predict_last_n/seq_length являются int, существует только одна частота
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}
            lowest_freq = sort_frequencies(ds.frequencies)[0]
            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue  # эта частота не прогнозируется
                results[basin][freq] = {}

                # масштабирование наблюдений
                # Получаем scaler для target_variables, используя правильное индексирование
                target_vars = self.cfg.target_variables
                feature_scaler = self.scaler["xarray_feature_scale"].sel(variable=target_vars).values
                feature_center = self.scaler["xarray_feature_center"].sel(variable=target_vars).values
                y_freq = y[freq] * feature_scaler + feature_center
                # масштабирование прогнозов
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    # если y_hat имеет 4 измерения, а у нас несколько признаков, мы расширяем измерения для масштабирования
                    feature_scaler = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                else:
                    raise RuntimeError(f"Симуляции имеют {y_hat[freq].ndim} измерений. Поддерживаются только 3 и 4.")

                # Создание словаря data_vars для xarray.Dataset
                data_vars = self._create_xarray_data_vars(y_hat_freq, y_freq)

                # freq_range - это шаги текущей частоты на каждом шаге самой низкой частоты
                frequency_factor = int(get_frequency_factor(lowest_freq, freq))

                # Создание словаря coords для xarray.Dataset. 'date' может быть напрямую выведен из словаря dates.
                # Мы индексируем выборку по дате последнего временного шага последовательности. Индекс 'time_step',
                # указывающий позицию в выходной последовательности (относительно конца), может быть выведен путем
                # вычисления разницы во времени дат. Чтобы учесть predict_last_n > 1 и многочастотность, нам нужно
                # добавить фактор частоты и вычесть 1 (чтобы начать с нуля).
                coords = {
                    'date':
                        dates[lowest_freq][:, -1],
                    'time_step': ((dates[freq][0, :] - dates[freq][0, -1]) / pd.Timedelta(freq)).astype(np.int64) +
                                 frequency_factor - 1
                }
                xr = xarray.Dataset(data_vars=data_vars, coords=coords)
                xr = xr.reindex({
                    'date':
                        pd.DatetimeIndex(pd.date_range(xr["date"].values[0], xr["date"].values[-1], freq=lowest_freq),
                                         name='date')
                })
                results[basin][freq]['xr'] = xr

                # создание диапазона дат на текущей частоте
                freq_date_range = pd.date_range(start=dates[lowest_freq][0, -1], end=dates[freq][-1, -1], freq=freq)
                # удаление временных шагов, которые не прогнозируются, из диапазона дат
                mask = np.ones(frequency_factor).astype(bool)
                mask[:-predict_last_n[freq]] = False
                freq_date_range = freq_date_range[np.tile(mask, len(xr['date']))]

                # только один раз предупреждать за частоту
                if frequency_factor < predict_last_n[freq] and basin == basins[0]:
                    tqdm.write(f'Метрики для {freq} рассчитываются только по последним {frequency_factor} элементам. '
                               f'Игнорируется {predict_last_n[freq] - frequency_factor} прогнозов на последовательность.')

                if metrics:
                    for target_variable in self.cfg.target_variables:
                        # объединение дат и временных шагов, чтобы не оценивать только каждые 24 часа при использовании use_frequencies=[1D, 1h]
                        obs = xr.isel(time_step=slice(-frequency_factor, None)) \
                            .stack(datetime=['date', 'time_step']) \
                            .drop_vars({'datetime', 'date', 'time_step'})[f"{target_variable}_obs"]
                        obs['datetime'] = freq_date_range
                        # проверка наличия наблюдений за этот период
                        if not all(obs.isnull()):
                            sim = xr.isel(time_step=slice(-frequency_factor, None)) \
                                .stack(datetime=['date', 'time_step']) \
                                .drop_vars({'datetime', 'date', 'time_step'})[f"{target_variable}_sim"]
                            sim['datetime'] = freq_date_range

                            # обрезка отрицательных прогнозов до нуля, если переменная указана в config 'clip_target_to_zero'
                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim = xarray.where(sim < 0, 0, sim)

                            if 'samples' in sim.dims:
                                sim = sim.mean(dim='samples')

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if 'all' in var_metrics:
                                var_metrics = get_available_metrics()
                            try:
                                values = calculate_metrics(obs, sim, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = f'Basin {basin} ' \
                                    + (f'{target_variable} ' if len(self.cfg.target_variables) > 1 else '') \
                                    + (f'{freq} ' if len(ds.frequencies) > 1 else '') \
                                    + str(err)
                                LOGGER.warning(msg)
                                values = {metric: np.nan for metric in var_metrics}

                            # добавление идентификатора переменной к метрикам, если необходимо
                            if len(self.cfg.target_variables) > 1:
                                values = {f"{target_variable}_{key}": val for key, val in values.items()}
                            # добавление идентификатора частоты к метрикам, если необходимо
                            if len(ds.frequencies) > 1:
                                values = {f"{key}_{freq}": val for key, val in values.items()}
                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)
                            for k, v in values.items():
                                results[basin][freq][k] = v

        # преобразование default dict обратно в обычный словарь Python, чтобы избежать неожиданного поведения при попытке доступа
        # к несуществующему бассейну
        results = dict(results)

        if (self.period == "validation") and (self.cfg.log_n_figures > 0) and (experiment_logger
                                                                               is not None) and results:
            self._create_and_log_figures(results, experiment_logger, epoch)

        # сохранение выходных данных модели в файл, если запрошено
        results_to_save = None
        states_to_save = None
        if save_results:
            results_to_save = results
        if save_all_output:
            states_to_save = all_output
        if save_results or save_all_output:
            self._save_results(results=results_to_save, states=states_to_save, epoch=epoch)

        return results

    def _create_and_log_figures(self, results: dict, experiment_logger: Logger, epoch: int):
        basins = list(results.keys())
        random.shuffle(basins)
        for target_var in self.cfg.target_variables:
            max_figures = min(self.cfg.validate_n_random_basins, self.cfg.log_n_figures, len(basins))
            for freq in results[basins[0]].keys():
                figures = []
                for i in range(max_figures):
                    xr = results[basins[i]][freq]['xr']
                    obs = xr[f"{target_var}_obs"].values
                    sim = xr[f"{target_var}_sim"].values
                    # обрезка отрицательных прогнозов до нуля, если переменная указана в config 'clip_target_to_zero'
                    if target_var in self.cfg.clip_targets_to_zero:
                        sim = xarray.where(sim < 0, 0, sim)
                    figures.append(
                        self._get_plots(
                            obs, sim, title=f"{target_var} - Basin {basins[i]} - Epoch {epoch} - Frequency {freq}")[0])
                # убедиться, что преамбула является допустимым именем файла
                experiment_logger.log_figures(figures, freq, preamble=re.sub(r"[^A-Za-z0-9\._\-]+", "", target_var))

    def _save_results(self, results: Optional[dict], states: Optional[dict] = None, epoch: int = None):
        """Сохранение результатов в различных форматах на диск.
        
        Примечание разработчика: Мы не можем сохранить временные ряды (объекты xarray) в виде файла netCDF, а должны использовать
        pickle в качестве обертки. Причина в том, что файлы netCDF имеют специальные ограничения на символы, которые могут
        использоваться в качестве имен переменных. Однако для удобства мы сохраним метрики, если они были рассчитаны, в отдельном csv-файле.
        """
        # использовать имя файла весов как часть имени папки с результатами
        weight_file = self._get_weight_file(epoch=epoch)

        # убедиться, что родительская директория существует
        parent_directory = self.run_dir / self.period / weight_file.stem
        parent_directory.mkdir(parents=True, exist_ok=True)

        # сохранять метрики в любое время, когда вызывается эта функция, при условии, что они существуют
        if self.cfg.metrics and results is not None:
            metrics_list = self.cfg.metrics
            if isinstance(metrics_list, dict):
                metrics_list = list(set(metrics_list.values()))
            if "all" in metrics_list:
                metrics_list = get_available_metrics()
            df = metrics_to_dataframe(results, metrics_list, self.cfg.target_variables)
            metrics_file = parent_directory / f"{self.period}_metrics.csv"
            df.to_csv(metrics_file)
            LOGGER.info(f"Метрики сохранены в {metrics_file}")

        # сохранить все результаты в pickle-файл
        if results is not None:
            result_file = parent_directory / f"{self.period}_results.p"
            with result_file.open("wb") as fp:
                pickle.dump(results, fp)
            LOGGER.info(f"Результаты сохранены в {result_file}")

        # сохранить все выходы модели в pickle-файл
        if states is not None:
            result_file = parent_directory / f"{self.period}_all_output.p"
            with result_file.open("wb") as fp:
                pickle.dump(states, fp)
            LOGGER.info(f"Состояния модели сохранены в {result_file}")

    def _evaluate(self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False):
        """Оценка модели"""
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {frequencies[0]: predict_last_n}  # если predict_last_n - это int, существует только одна частота

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []
        with torch.no_grad():
            for data in loader:

                for key in data:
                    if key.startswith('x_d'):
                        data[key] = {k: v.to(self.device, non_blocking=True) for k, v in data[key].items()}
                    elif not key.startswith('date'):
                        data[key] = data[key].to(self.device, non_blocking=True)
                data = model.pre_model_hook(data, is_train=False)
                predictions, loss = self._get_predictions_and_loss(model, data)

                if all_output:
                    for key, value in predictions.items():
                        if value is not None and type(value) != dict:
                            all_output[key].append(value.detach().cpu().numpy())
                elif save_all_output:
                    all_output = {
                        key: [value.detach().cpu().numpy()]
                        for key, value in predictions.items()
                        if value is not None and type(value) != dict
                    }

                for freq in frequencies:
                    if predict_last_n[freq] == 0:
                        continue  # нет прогнозов для этой частоты
                    freq_key = '' if len(frequencies) == 1 else f'_{freq}'
                    y_hat_sub, y_sub = self._subset_targets(model, data, predictions, predict_last_n[freq], freq_key)
                    # обрезка дат универсальна для всех моделей и происходит здесь
                    date_sub = data[f'date{freq_key}'][:, -predict_last_n[freq]:]

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                        dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

                losses.append(loss)

            for freq in preds.keys():
                preds[freq] = preds[freq].numpy()
                obs[freq] = obs[freq].numpy()

        # объединить все выходные переменные (в настоящее время dict-of-dicts) в dict одного уровня
        for key, list_of_data in all_output.items():
            all_output[key] = np.concatenate(list_of_data, 0)

        # установить явно NaN, если все потери NaN, чтобы избежать RuntimeWarning
        mean_losses = {}
        if len(losses) == 0:
            mean_losses['loss'] = np.nan
        else:
            for loss_name in losses[0].keys():
                loss_values = [loss[loss_name] for loss in losses]
                mean_losses[loss_name] = np.nanmean(loss_values) if not np.all(np.isnan(loss_values)) else np.nan

        return preds, obs, dates, mean_losses, all_output

    def _get_predictions_and_loss(self, model: BaseModel, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        predictions = model(data)
        _, all_losses = self.loss_obj(predictions, data)
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: np.ndarray,
                        predict_last_n: int, freq: str):
        raise NotImplementedError

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        raise NotImplementedError


class RegressionTester(BaseTester):
    """Класс-тестер для запуска инференса на регрессионной модели.

    Используйте метод `evaluate` этого класса для оценки обученной модели на ее обучающем, тестовом или валидационном периоде.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    run_dir : Path
        Путь к директории запуска.
    period : {'train', 'validation', 'test'}
        Период для оценки.
    init_model : bool, optional
        Если True, веса модели будут инициализированы контрольной точкой из последней доступной эпохи в `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(RegressionTester, self).__init__(cfg, run_dir, period, init_model)

    def _subset_targets(self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: np.ndarray,
                        predict_last_n: np.ndarray, freq: str):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (('date', 'time_step'), y[:, :, i])
            data[f"{var}_sim"] = (('date', 'time_step'), y_hat[:, :, i])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.regression_plot(qobs, qsim, title)


class UncertaintyTester(BaseTester):
    """Класс-тестер для запуска инференса на модели неопределенности.

    Используйте метод `evaluate` этого класса для оценки обученной модели на ее обучающем, тестовом или валидационном периоде.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    run_dir : Path
        Путь к директории запуска.
    period : {'train', 'validation', 'test'}
        Период для оценки.
    init_model : bool, optional
        Если True, веса модели будут инициализированы контрольной точкой из последней доступной эпохи в `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(UncertaintyTester, self).__init__(cfg, run_dir, period, init_model)

    def _get_predictions_and_loss(self, model: BaseModel, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        outputs = model(data)
        _, all_losses = self.loss_obj(outputs, data)
        predictions = model.sample(data, self.cfg.n_samples)
        model.eval()
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(self,
                        model: BaseModel,
                        data: Dict[str, torch.Tensor],
                        predictions: np.ndarray,
                        predict_last_n: int,
                        freq: str = None):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (('date', 'time_step'), y[:, :, i])
            data[f"{var}_sim"] = (('date', 'time_step', 'samples'), y_hat[:, :, i, :])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.uncertainty_plot(qobs, qsim, title)
