import itertools
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from ruamel.yaml import YAML


def load_basin_id_encoding(run_dir: Path) -> Dict[str, int]:
    id_to_int_file = run_dir / "train_data" / "id_to_int.yml"
    if id_to_int_file.is_file():
        with id_to_int_file.open("r") as fp:
            yaml = YAML(typ="safe")
            id_to_int = yaml.load(fp)
        return id_to_int

    else:
        id_to_int_file = run_dir / "train_data" / "id_to_int.p"
        if id_to_int_file.is_file():
            with id_to_int_file.open("rb") as fp:
                id_to_int = pickle.load(fp)
            return id_to_int
        else:
            raise FileNotFoundError(f"Файл id-to-int не найден в {id_to_int_file.parent}. "
                                    "Искали (новый) yaml файл или (старый) pickle файл")



def metrics_to_dataframe(results: dict, metrics: Iterable[str], targets: Iterable[str]) -> pd.DataFrame:
    """Извлечь все значения метрик из результирующего словаря и преобразовать в pandas.DataFrame

    Параметры
    ----------
    results : dict
        Словарь, содержащий результаты оценки модели, возвращаемые `Tester.evaluate()`.
    metrics : Iterable[str]
        Итерируемый объект с именами метрик (без суффикса частоты).
    targets : Iterable[str]
        Итерируемый объект с именами целевых переменных.

    Возвращает
    -------
    Индексированный DataFrame с одним столбцом на каждую метрику. В случае многочастотных запусков названия метрик
    содержат соответствующий суффикс частоты.
    """
    metrics_dict = defaultdict(dict)
    for basin, basin_data in results.items():
        for freq, freq_results in basin_data.items():
            for target, metric in itertools.product(targets, metrics):
                metric_key = metric
                if len(targets) > 1:
                    metric_key = f"{target}_{metric}"
                if len(basin_data) > 1:
                    # для многочастотных запусков метрики включают суффикс частоты
                    metric_key = f"{metric_key}_{freq}"
                if metric_key in freq_results.keys():
                    metrics_dict[basin][metric_key] = freq_results[metric_key]
                else:
                    # в случае, если текущий период не имеет допустимых выборок, результирующий словарь не имеет metric-key
                    metrics_dict[basin][metric_key] = np.nan

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.index.name = "basin"

    return df
