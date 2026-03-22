"""Скрипт утилиты для генерации конфигурационных файлов на основе базовой конфигурации и определенного набора вариаций."""
import itertools
from pathlib import Path
from typing import Dict

from neuralprognoz.utils.config import Config


def create_config_files(base_config_path: Path, modify_dict: Dict[str, list], output_dir: Path):
    """Создает файлы конфигурации на основе базового файла конфигурации и словаря параметров для модификации.
    
    Данный скрипт создаст отдельный файл конфигурации для каждой комбинации параметров, указанных в modify_dict.
    
    Параметры
    ----------
    base_config_path : Path
        Путь к основному файлу конфигурации (.yml)
    modify_dict : dict
        Словарь, сопоставляющий имена параметров со списком возможных значений параметров.
    output_dir : Path 
        Путь к каталогу, куда будут помещены созданные файлы конфигурации.
    """
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # загрузка базовой конфигурации
    base_config = Config(base_config_path)
    experiment_name = base_config.experiment_name
    option_names = list(modify_dict.keys())

    # Итерироваться по каждой возможной комбинации гиперпараметров.
    for i, options in enumerate(itertools.product(*[val for val in modify_dict.values()])):

        base_config.update_config(dict(zip(option_names, options)))

        # создать уникальное название эксперимента
        name = experiment_name
        for key, val in zip(option_names, options):
            name += f"_{key}{val}"
        base_config.update_config({"experiment_name": name})

        base_config.dump_config(output_dir, f"config_{i+1}.yml")

    print(f"Завершено. Конфигурации сохранены в {output_dir}")
