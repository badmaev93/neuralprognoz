"""Скрипт для осреднения прогнозов нескольких запусков. """
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from neuralprognoz.datautils.utils import get_frequency_factor, sort_frequencies
from neuralprognoz.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralprognoz.evaluation.utils import metrics_to_dataframe
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.errors import AllNaNError


def create_results_ensemble(run_dirs: List[Path],
                            best_k: int = None,
                            metrics: List[str] = None,
                            period: str = 'test',
                            epoch: int = None) -> dict:
    """Вычисление среднего значения прогнозов нескольких запусков за указанный период и рассчет новых метрик.

    Если указан параметр `best_k`, то будут выбраны только k прогонов с наилучшим показателем NSE на этапе валидации.
    
    Параметры
    ----------
    run_dirs : List[Path]
        Список директорий запусков, которые подлежат слиянию.
    best_k : int, необязательный
        Если указан, то объединение будет выполнено только для k лучших запусков на основании показателя NSE на этапе валидации.
    metrics : List[str], необязательный
        Использовать этот параметр для переопределения метрик из конфигурационных файлов в директориях запусков.
    period : {'test', 'validation', 'train'}, необязательный
        Период: одно из значений train, val, test. Если используется best_k, разрешается только 'test'. 
        Директории прогонов должны содержать файлы результатов для указанного периода.
    epoch : int, необязательный
        Если указан, ансамбль будет построен на основе прогнозов модели за эту эпоху, иначе - за последнюю эпоху.
    
    Возвращает
    -------
    dict
        Словарь с прогнозами ансамбля и метриками по бассейнам и частотам.
    """
    if len(run_dirs) < 2:
        raise ValueError('Необходимо предоставить как минимум две директории запусков для объединения.')

    if period not in ['train', 'validation', 'test']:
        raise ValueError(f'Неизвестный период {period}.')
    if best_k is not None:
        if period != 'test':
            raise ValueError('Если указан параметр `best_k`, то период должен быть тестовым (`test`).')
        print('Поиск запусков с лучшими показателями на этапе валидации.')
        best_val_runs = _get_best_validation_runs(run_dirs, best_k, epoch)
        best_runs = [_get_results_file(run_dir, period, epoch) for run_dir in best_val_runs]
    else:
        best_runs = [_get_results_file(run_dir, period, epoch) for run_dir in run_dirs]

    config = Config(run_dirs[0] / 'config.yml')
    if metrics is not None:
        # переопределить метрики из конфигурации
        config.metrics = metrics

    # получить частоты из файла результатов
    # (они могут отсутствовать в конфигурации, если использовалась нативная частота данных)
    run_results = pickle.load(open(best_runs[0], 'rb'))
    frequencies = list(run_results[list(run_results.keys())[0]].keys())

    return _create_ensemble(best_runs, frequencies, config)


def _create_ensemble(results_files: List[Path], frequencies: List[str], config: Config) -> dict:
    """Вычисляет среднее значение прогнозов прошедших запусков и повторно рассчитывает метрики. """
    lowest_freq = sort_frequencies(frequencies)[0]
    ensemble_sum = defaultdict(dict)
    target_vars = config.target_variables

    print('Загрузка результатов для каждого запуска')
    for run in tqdm(results_files):
        run_results = pickle.load(open(run, 'rb'))
        for basin, basin_results in run_results.items():
            for freq in frequencies:
                freq_results = basin_results[freq]['xr']

                # Суммирует прогнозы по всем бассейнам
                if freq not in ensemble_sum[basin]:
                    ensemble_sum[basin][freq] = freq_results
                else:
                    for target_var in target_vars:
                        ensemble_sum[basin][freq][f'{target_var}_sim'] += freq_results[f'{target_var}_sim']

    # Разделить сумму прогнозов на количество запусков, чтобы получить среднее прогнозное значение для каждого бассейна и частоты
    print('Объединие результатов и рассчет метрик.')
    ensemble = defaultdict(lambda: defaultdict(dict))
    for basin in tqdm(ensemble_sum.keys()):
        for freq in frequencies:
            ensemble_xr = ensemble_sum[basin][freq]

            # объединить дату и время в единый индекс для расчёта метрик
            # создать диапазон дат-времени на текущей частоте, удаляя временные шаги, которые не прогнозируются
            frequency_factor = int(get_frequency_factor(lowest_freq, freq))
            # убедиться, что последний день полностью входит в указанный диапазон.
            freq_date_range = pd.date_range(start=ensemble_xr.coords['date'].values[0],
                                            end=ensemble_xr.coords['date'].values[-1] \
                                                + pd.Timedelta(days=1, seconds=-1),
                                            freq=freq)
            mask = np.ones(frequency_factor).astype(bool)
            mask[:-len(ensemble_xr.coords['time_step'])] = False
            freq_date_range = freq_date_range[np.tile(mask, len(ensemble_xr.coords['date']))]

            ensemble_xr = ensemble_xr.isel(time_step=slice(-frequency_factor, None)) \
                .stack(datetime=['date', 'time_step']) \
                .drop_vars({'datetime', 'date', 'time_step'})
            ensemble_xr['datetime'] = freq_date_range
            for target_var in target_vars:
                # средний прогноз
                ensemble_xr[f'{target_var}_sim'] = ensemble_xr[f'{target_var}_sim'] / len(results_files)

                # обрезать прогнозы до нуля.
                sim = ensemble_xr[f'{target_var}_sim']
                if target_var in config.clip_targets_to_zero:
                    sim = xr.where(sim < 0, 0, sim)

                # рассчет метрик
                metrics = config.metrics if isinstance(config.metrics, list) else config.metrics[target_var]
                if 'all' in metrics:
                    metrics = get_available_metrics()
                try:
                    ensemble_metrics = calculate_metrics(ensemble_xr[f'{target_var}_obs'],
                                                         sim,
                                                         metrics=metrics,
                                                         resolution=freq)
                except AllNaNError as err:
                    msg = f'Бассейн {basin} ' \
                        + (f'{target_var} ' if len(target_vars) > 1 else '') \
                        + (f'{freq} ' if len(frequencies) > 1 else '') \
                        + str(err)
                    print(msg)
                    ensemble_metrics = {metric: np.nan for metric in metrics}

                # добавить идентификатор переменной к метрикам, если нужно
                if len(target_vars) > 1:
                    ensemble_metrics = {f'{target_var}_{key}': val for key, val in ensemble_metrics.items()}
                # добавить идентификатор частоты к метрикам, если нужно
                if len(frequencies) > 1:
                    ensemble_metrics = {f'{key}_{freq}': val for key, val in ensemble_metrics.items()}
                for metric, val in ensemble_metrics.items():
                    ensemble[basin][freq][metric] = val

            ensemble[basin][freq]['xr'] = ensemble_xr

    return dict(ensemble)


def _get_medians(results: dict, metric='NSE') -> dict:
    """Рассчет медианных метрик по всем бассейнам. """
    medians = {}
    key = metric
    frequencies = list(results[list(results.keys())[0]].keys())
    for freq in frequencies:
        if len(frequencies) > 1:
            key = f'{metric}_{freq}'
        metric_values = [v[freq][key] for v in results.values() if freq in v.keys() and key in v[freq].keys()]
        medians[freq] = np.nanmedian(metric_values)

    return medians


def _get_best_validation_runs(run_dirs: List[Path], k: int, epoch: int = None) -> List[Path]:
    """Возвращает k директорий прогонов с наилучшими средними показателями на этапе валидации. """
    val_files = list(zip(run_dirs, [_get_results_file(run_dir, 'validation', epoch) for run_dir in run_dirs]))

    # получить медианные значения валидации
    median_sums = {}
    for run_dir, val_file in val_files:
        val_results = pickle.load(open(val_file, 'rb'))
        val_medians = _get_medians(val_results)
        print('validation', val_file, val_medians)
        median_sums[run_dir] = sum(val_medians.values())

    if k > len(run_dirs):
        raise ValueError(f'best_k k больше чем количество запусков {len(val_files)}.')
    return sorted(median_sums, key=median_sums.get, reverse=True)[:k]


def _get_results_file(run_dir: Path, period: str = 'test', epoch: int = None) -> Path:
    """Возвращает путь к файлу результатов в заданной директории запуска. """
    if epoch is not None:
        dir_results_files = list(Path(run_dir).glob(f'{period}/model_epoch{str(epoch).zfill(3)}/{period}_results.p'))
    else:
        dir_results_files = list(Path(run_dir).glob(f'{period}/model_epoch*/{period}_results.p'))
    if len(dir_results_files) == 0:
        raise ValueError(f'{run_dir}  не содержит результаты за период {period}.')
    return sorted(dir_results_files)[-1]

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', type=str, nargs='+', help='Каталоги запусков, прогнозы которых подлежат усреднению.')
    parser.add_argument('--period', type=str, choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--output-dir', type=str, help='Путь к каталогу, в котором хранятся результаты.')
    parser.add_argument('--metrics',
                        type=str,
                        nargs='+',
                        required=False,
                        help='Опция для переопределения метрик из конфигурации.')
    parser.add_argument('--best-k',
                        type=int,
                        required=False,
                        help='Если указан, будут использованы только k файлов результатов с наилучшими медианными значениями NSE на этапе валидации.')
    parser.add_argument('--epoch',
                        type=int,
                        required=False,
                        help='Если указано, возвращает результаты для конкретной эпохи, в противном случае - для последней эпохи.')
    args = vars(parser.parse_args())

    run_dirs = [Path(f) for f in args['run_dirs']]
    ensemble_results = create_results_ensemble(run_dirs,
                                               args['best_k'],
                                               metrics=args['metrics'],
                                               period=args['period'],
                                               epoch=args['epoch'])
    output_dir = Path(args['output_dir']).absolute()

    metrics = args['metrics']
    cfg = Config(run_dirs[0] / 'config.yml')
    if metrics is None:
        metrics = cfg.metrics
        if isinstance(metrics, dict):
            metrics = list(set(metrics.values()))
    if 'all' in metrics:
        metrics = get_available_metrics()
    try:
        df = metrics_to_dataframe(ensemble_results, metrics, cfg.target_variables)
        file_name = output_dir / f"{args['period']}_ensemble_metrics.csv"
        df.to_csv(file_name)
        print(f"Метрики ансамбля запусков были сохранены в... {file_name}")
    except RuntimeError as err:
        # если метрики не были рассчитаны
        pass

    file_name = output_dir / f"{args['period']}_ensemble_results.p"
    pickle.dump(ensemble_results, open(file_name, 'wb'))
    print(f'Результаты успешно записаны в... {file_name}')


if __name__ == '__main__':
    _main()
