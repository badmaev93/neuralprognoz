import itertools
import random
import re
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from ruamel.yaml import YAML


class Config(object):
    """Читает конфигурацию запуска из указанного пути или словаря и преобразует её в объект конфигурации.

    Во время обработки ключи конфигурации, содержащие 'dir', 'file' или 'path', будут преобразованы в экземпляры pathlib.Path.
    Ключ конфигурации, заканчивающийся на '_date', будет интерпретирован как pd.Timestamp. Ожидаемый формат: ДД/ММ/ГГГГ.

    Параметры
    ----------
    yml_path_or_dict : Union[Path, dict]
        Путь к файлу конфигурации или словарь с параметрами конфигурации.
    dev_mode : bool, optional
        Если режим разработки отключен, создание конфигурации завершится неудачей, если среди переданных параметров имеются 
        неизвестные ключи.
        Режим разработки можно активировать либо через этот параметр, либо установив в спецификации конфигурации ``dev_mode: True`` 
        в `yml_path_or_dict`.

    Вызывает
    ------
    ValueError
        Если переданная спецификация конфигурации не является ни путём, ни словарем, или если режим разработки выключен и файл 
        или словарь содержат незнакомые ключи.
    """

    # Определение списков устаревших и чисто информативных ключей конфигурации
    _deprecated_keys = [
        'static_inputs', 'camels_attributes', 'target_variable', 'embedding_hiddens', 'embedding_activation',
        'embedding_dropout'
    ]
    _metadata_keys = ['package_version', 'commit_hash']

    def __init__(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        if isinstance(yml_path_or_dict, Path):
            self._cfg = Config._read_and_parse_config(yml_path=yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            self._cfg = Config._parse_config(yml_path_or_dict)
        else:
            raise ValueError(f'Не возможно создать конфигурацию из входного файла типа {type(yml_path_or_dict)}.')

        if not (self._cfg.get('dev_mode', False) or dev_mode):
            Config._check_cfg_keys(self._cfg)

        # Настройка названия эксперимента
        if "experiment_name" in self._cfg and self._cfg["experiment_name"]:
            # 1. Заменить фигурные скобки на нужные значения.
            new_name = self._cfg["experiment_name"]
            for key, val in self._cfg.items():
                if key.endswith('_date'):
                    if isinstance(val, list):
                        date_string = '_'.join(elem.strftime('%Y-%m-%d') for elem in val)
                    else:
                        date_string = val.strftime('%Y-%m-%d')
                    new_name = re.sub(f'{{{key}}}', date_string, new_name)
            new_name = re.sub('{random_name}', create_random_name(), new_name)
            try:
                new_name = new_name.format(**self._cfg)
            except KeyError as ex:
                raise KeyError(f'Название эксперимента - {self._cfg["experiment_name"]} ' +
                               f'но {{{ex}}} не найден в конфигурации') from ex
            # 2. Удалить фигурные скобки и убедиться, что название эксперимента подходит для использования в качестве папки в Linux и Windows. 
            new_name = re.sub('{', "(", new_name)
            new_name = re.sub('}', ")", new_name)
            new_name = re.sub('"', "'", new_name)
            new_name = re.sub('[<>:"/\\\\|?*]', "_", new_name)
            new_name = re.sub(' ', '', new_name)

            self._cfg["experiment_name"] = new_name

    def as_dict(self) -> dict:
        """Возвращает конфигурацию запуска в виде словаря.
        
        Возвращает
        -------
        dict
            Конфигурация запуска, определённая в файле .yml.
        """
        return self._cfg

    def dump_config(self, folder: Path, filename: str = 'config.yml'):
        """"Сохраняет конфигурацию запуска в виде файла .yml на диск.

        Параметры
        ----------
        folder : Path
            Папка, в которой будет сохранена конфигурация.
        filename : str, optional
            Имя файла, который будет сохранён. По умолчанию: 'config.yml'.

        Вызывает
        ------
        FileExistsError
            Если в указанной папке уже существует файл с именем `filename`.
        """
        yml_path = folder / filename
        if not yml_path.exists():
            with yml_path.open('w') as fp:
                temp_cfg = {}
                for key, val in self._cfg.items():
                    if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(str(elem))
                            temp_cfg[key] = temp_list
                        else:
                            temp_cfg[key] = str(val)
                    elif key.endswith('_date'):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(elem.strftime(format="%d/%m/%Y"))
                            temp_cfg[key] = temp_list
                        else:
                            # Игнорировать None вызванные, например использованием отдельного per_basin_period_file
                            if isinstance(val, pd.Timestamp):
                                temp_cfg[key] = val.strftime(format="%d/%m/%Y")
                    else:
                        temp_cfg[key] = val

                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            raise FileExistsError(yml_path)

    def update_config(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        """Обновляет аргументы конфигурации.
        
        Полезно, например, в контексте тонкой настройки или продолжения тренировки с контрольной точки для адаптации, 
        например, скорости обучения, файлов бассейнов или любых других параметров.
        
        Параметры
        ----------
        yml_path_or_dict : Union[Path, dict]
            Путь к новому файлу конфигурации или словарь с параметрами конфигурации. Каждый аргумент, указанный 
            в данном файле, перезапишет существующий аргумент конфигурации.
        dev_mode : bool, optional
            Если режим разработки отключен, процесс обновления конфигурации завершится неудачно, если в переданном 
            файле конфигурации указаны непознаваемые ключи. Режим разработки можно включить либо через этот параметр, 
            либо задав ``dev_mode: True`` в `yml_path_or_dict`.

        Вызывает
        ------
        ValueError
            Если спецификация конфигурации не является ни путём, ни словарём, или если режим разработки отключен 
            (по умолчанию), а файл или словарь содержат незнакомые ключи.
        """
        new_config = Config(yml_path_or_dict, dev_mode=dev_mode)

        self._cfg.update(new_config.as_dict())

    def _get_value_verbose(self, key: str) -> Union[float, int, str, list, dict, Path, pd.Timestamp]:
        """Использование этой функции внутренне для возврата обязательных атрибутов конфигурации."""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} не специфицирован config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} является обязательным, но в конфигурации указан как 'None'.")
        else:
            return self._cfg[key]

    @staticmethod
    def _as_default_list(value: Any) -> list:
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    def _as_default_dict(value: Any) -> dict:
        if value is None:
            return {}
        elif isinstance(value, dict):
            return value
        else:
            raise RuntimeError(f"Несовместимый тип {type(value)}. Ожидается `dict` или `None`.")

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        """Проверка конфигурации на неизвестные ключи. """
        property_names = [p for p in dir(Config) if isinstance(getattr(Config, p), property)]

        unknown_keys = [
            k for k in cfg.keys()
            if k not in property_names and k not in Config._deprecated_keys and k not in Config._metadata_keys
        ]
        if unknown_keys:
            raise ValueError(f'{unknown_keys} - нераспознанные ключи конфигурации')

    @staticmethod
    def _parse_config(cfg: dict) -> dict:
        for key, val in cfg.items():
            # преобразование всех строковых путей в объекты PosixPath
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # преобразование Dates в pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Проверка длины последовательности прогноза.
        if cfg.get('forecast_seq_length'):
            if cfg['forecast_seq_length'] >= cfg['seq_length']:
                raise ValueError('длина последовательности прогноза должна быть < длины последовательности.')
            if cfg.get('forecast_overlap'):
                if cfg['forecast_overlap'] > cfg['forecast_seq_length']:
                    raise ValueError('Перехлест прогноза должен быть  <=  длине последовательности прогноза.')

        # Проверка авторегрессионныз входов.
        if 'autoregressive_inputs' in cfg:
            if len(cfg['autoregressive_inputs']) > 1:
                raise ValueError('Пока что поддерживается только один авторегресионный вход.')
            if cfg['autoregressive_inputs'] and len(cfg['target_variables']) > 1:
                raise ValueError('Пока что авторегресионные модели поддерживают только одну целевую переменную.')
            if not cfg['autoregressive_inputs'][0].startswith(cfg['target_variables'][0]):
                raise ValueError('Авторегресионный вход должен быть лаговой версией целевой переменной.')

        # Добавлениепарсинга дополнительных конфигураций, если будет нужно
        return cfg

    @staticmethod
    def _read_and_parse_config(yml_path: Path):
        if yml_path.exists():
            with yml_path.open('r', encoding='utf-8') as fp:
                yaml = YAML(typ="safe")
                cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(yml_path)
        cfg = Config._parse_config(cfg)

        return cfg

    @property
    def additional_feature_files(self) -> List[Path]:
        return self._as_default_list(self._cfg.get("additional_feature_files", None))

    @property
    def allow_subsequent_nan_losses(self) -> int:
        return self._cfg.get("allow_subsequent_nan_losses", 0)

    @property
    def autoregressive_inputs(self) -> Union[List[str], Dict[str, List[str]]]:
        return self._as_default_list(self._cfg.get("autoregressive_inputs", []))

    @property
    def base_run_dir(self) -> Path:
        return self._get_value_verbose("base_run_dir")

    @base_run_dir.setter
    def base_run_dir(self, folder: Path):
        self._cfg["base_run_dir"] = folder

    @property
    def batch_size(self) -> int:
        return self._get_value_verbose("batch_size")

    @property
    def bidirectional_stacked_forecast_lstm(self) -> bool:
        return self._cfg.get("bidirectional_stacked_forecast_lstm", False)

    @property
    def cache_validation_data(self) -> bool:
        return self._cfg.get("cache_validation_data", True)

    @property
    def checkpoint_path(self) -> Path:
        return self._cfg.get("checkpoint_path", None)

    @property
    def clip_gradient_norm(self) -> float:
        return self._cfg.get("clip_gradient_norm", None)

    @property
    def clip_targets_to_zero(self) -> List[str]:
        return self._as_default_list(self._cfg.get("clip_targets_to_zero", []))

    @property
    def continue_from_epoch(self) -> int:
        return self._cfg.get("continue_from_epoch", None)

    @property
    def custom_normalization(self) -> dict:
        return self._as_default_dict(self._cfg.get("custom_normalization", {}))

    @property
    def data_dir(self) -> Path:
        return self._get_value_verbose("data_dir")

    @property
    def dataset(self) -> str:
        return self._get_value_verbose("dataset")

    @property
    def device(self) -> str:
        return self._cfg.get("device", None)

    @device.setter
    def device(self, device: str):
        if device == "cpu" or device.startswith("cuda:") or device == "mps":
            self._cfg["device"] = device
        else:
            raise ValueError("Параметр 'device' должен быть одним из следующих: 'cpu', 'mps', or 'cuda:X', где 'X' - GPU ID.")

    @property
    def duplicate_features(self) -> dict:
        duplicate_features = self._cfg.get("duplicate_features", {})
        if duplicate_features is None:
            return {}
        elif isinstance(duplicate_features, dict):
            return duplicate_features
        elif isinstance(duplicate_features, list):
            return {feature: 1 for feature in duplicate_features}
        elif isinstance(duplicate_features, str):
            return {duplicate_features: 1}
        else:
            raise RuntimeError(f"Неподдерживаемый тип {type(duplicate_features)} для аргумента 'duplicate_features'.")

    @property
    def dynamic_conceptual_inputs(self) -> List[str]:
        return self._as_default_list(self._cfg.get("dynamic_conceptual_inputs", []))

    @property
    def warmup_period(self) -> int:
        return self._cfg.get("warmup_period", 0)

    @property
    def dynamic_inputs(self) -> Union[list[str], list[list[str]], dict[str, list[str]]]:
        return self._get_value_verbose("dynamic_inputs")

    @property
    def dynamic_inputs_flattened(self) -> list[str]:
        dynamic_inputs = self.dynamic_inputs
        if isinstance(dynamic_inputs, dict):
            return list(itertools.chain.from_iterable(dynamic_inputs.values()))
        if dynamic_inputs and isinstance(dynamic_inputs[0], list):
            return list(itertools.chain.from_iterable(dynamic_inputs))
        else:
            return dynamic_inputs

    @property
    def dynamics_embedding(self) -> dict:
        embedding_spec = self._cfg.get("dynamics_embedding", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def epochs(self) -> int:
        return self._get_value_verbose("epochs")

    @property
    def evolving_attributes(self) -> List[str]:
        if "evolving_attributes" in self._cfg.keys():
            return self._as_default_list(self._cfg["evolving_attributes"])
        elif "static_inputs" in self._cfg.keys():
            warnings.warn("Использование ключа 'static_inputs' устарело. Используйте в будущем 'evolving_attributes'", FutureWarning)
            return self._as_default_list(self._cfg["static_inputs"])
        else:
            return []

    @property
    def experiment_name(self) -> str:
        if self._cfg.get("experiment_name", None) is None:
            return "run"
        else:
            return self._cfg["experiment_name"]

    @property
    def finetune_modules(self) -> Union[List[str], Dict[str, str]]:
        finetune_modules = self._cfg.get("finetune_modules", [])
        if finetune_modules is None:
            return []
        elif isinstance(finetune_modules, str):
            return [finetune_modules]
        elif isinstance(finetune_modules, dict) or isinstance(finetune_modules, list):
            return finetune_modules
        else:
            raise ValueError(f"Неизвестный тип данных {type(finetune_modules)} для аргумента 'finetune_modules'.")

    @property
    def forecast_network(self) -> dict:
        embedding_spec = self._cfg.get("forecast_network", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def forecast_hidden_size(self) -> int:
        return self._cfg.get("forecast_hidden_size", self.hidden_size)

    @property
    def forecast_inputs(self) -> list[str] | list[list[str]]:
        return self._cfg.get("forecast_inputs", [])

    @property
    def forecast_inputs_flattened(self) -> list[str]:
        forecast_inputs = self.forecast_inputs
        if forecast_inputs and isinstance(forecast_inputs[0], list):
            return list(itertools.chain.from_iterable(forecast_inputs))
        else:
            return forecast_inputs

    @property
    def forecast_overlap(self) -> int:
        return self._cfg.get("forecast_overlap", None)

    @property
    def forecast_seq_length(self) -> int:
        return self._cfg.get("forecast_seq_length", None)

    @property
    def forcings(self) -> List[str]:
        return self._as_default_list(self._get_value_verbose("forcings"))

    @property
    def save_git_diff(self) -> bool:
        return self._cfg.get('save_git_diff', False)

    @property
    def state_handoff_network(self) -> dict:
        embedding_spec = self._cfg.get("state_handoff_network", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def head(self) -> str:
        if self.model == "mclstm":
            return ''
        else:
            return self._get_value_verbose("head")

    @property
    def hindcast_inputs(self) -> list[str] | list[list[str]]:
        return self._cfg.get("hindcast_inputs", [])

    @property
    def hindcast_inputs_flattened(self) -> list[str]:
        hindcast_inputs = self.hindcast_inputs
        if hindcast_inputs and isinstance(hindcast_inputs[0], list):
            return list(itertools.chain.from_iterable(hindcast_inputs))
        else:
            return hindcast_inputs

    @property
    def hidden_size(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("hidden_size")

    @property
    def hindcast_hidden_size(self) -> Union[int, Dict[str, int]]:
        return self._cfg.get("hindcast_hidden_size", self.hidden_size)

    @property
    def hydroatlas_attributes(self) -> List[str]:
        return self._as_default_list(self._cfg.get("hydroatlas_attributes", []))

    @property
    def img_log_dir(self) -> Path:
        return self._cfg.get("img_log_dir", None)

    @img_log_dir.setter
    def img_log_dir(self, folder: Path):
        self._cfg["img_log_dir"] = folder

    @property
    def initial_forget_bias(self) -> float:
        return self._cfg.get("initial_forget_bias", None)

    @property
    def is_continue_training(self) -> bool:
        return self._cfg.get("is_continue_training", False)

    @is_continue_training.setter
    def is_continue_training(self, flag: bool):
        self._cfg["is_continue_training"] = flag

    @property
    def is_finetuning(self) -> bool:
        return self._cfg.get("is_finetuning", False)

    @is_finetuning.setter
    def is_finetuning(self, flag: bool):
        self._cfg["is_finetuning"] = flag

    @property
    def lagged_features(self) -> dict:
        return self._as_default_dict(self._cfg.get("lagged_features", {}))

    @property
    def learning_rate(self) -> Dict[int, float]:
        if ("learning_rate" in self._cfg.keys()) and (self._cfg["learning_rate"] is not None):
            if isinstance(self._cfg["learning_rate"], float):
                return {0: self._cfg["learning_rate"]}
            elif isinstance(self._cfg["learning_rate"], dict):
                return self._cfg["learning_rate"]
            else:
                raise ValueError("Неверный тип данных для скорости обучения. Используйте либо словарь (эпоха => float) или float.")
        else:
            raise ValueError("Скорость обучения не указана в конфигурации (.yml-файл).")

    @property
    def log_interval(self) -> int:
        return self._cfg.get("log_interval", 10)

    @property
    def log_n_figures(self) -> int:
        if (self._cfg.get("log_n_figures", None) is None) or (self._cfg["log_n_figures"] < 1):
            return 0
        else:
            return self._cfg["log_n_figures"]

    @property
    def log_tensorboard(self) -> bool:
        return self._cfg.get("log_tensorboard", True)

    @property
    def loss(self) -> str:
        return self._get_value_verbose("loss")

    @loss.setter
    def loss(self, loss: str):
        self._cfg["loss"] = loss

    @property
    def mamba_d_conv(self) -> int:
        return self._cfg.get("d_conv", 4)

    @property
    def mamba_d_state(self) -> int:
        return self._cfg.get("d_state", 16)

    @property
    def mamba_expand(self) -> int:
        return self._cfg.get("expand", 2)

    @property
    def mass_inputs(self) -> List[str]:
        return self._as_default_list(self._cfg.get("mass_inputs", []))

    @property
    def max_updates_per_epoch(self) -> Optional[int]:
        return self._cfg.get("max_updates_per_epoch")

    @property
    def mc_dropout(self) -> bool:
        return self._cfg.get("mc_dropout", False)

    @property
    def metrics(self) -> Union[List[str], Dict[str, List[str]]]:
        return self._cfg.get("metrics", [])

    @metrics.setter
    def metrics(self, metrics: Union[str, List[str], Dict[str, List[str]]]):
        self._cfg["metrics"] = metrics

    @property
    def model(self) -> str:
        return self._get_value_verbose("model")

    @property
    def conceptual_model(self) -> str:
        return self._cfg.get("conceptual_model", "SHM")

    @property
    def n_distributions(self) -> int:
        return self._get_value_verbose("n_distributions")

    @property
    def n_samples(self) -> int:
        return self._get_value_verbose("n_samples")

    @property
    def n_taus(self) -> int:
        return self._get_value_verbose("n_taus")

    @property
    def nan_handling_method(self) -> str:
        method = self._cfg.get("nan_handling_method", None)
        if not method:
            return None
        return method

    @property
    def nan_sequence_probability(self) -> float:
        return self._cfg.get("nan_sequence_probability", 0.0)

    @property
    def nan_step_probability(self) -> float:
        return self._cfg.get("nan_step_probability", 0.0)

    @property
    def nan_handling_pos_encoding_size(self) -> int:
        return self._cfg.get("nan_handling_pos_encoding_size", 0)

    @property
    def negative_sample_handling(self) -> str:
        return self._cfg.get("negative_sample_handling", None)

    @property
    def negative_sample_max_retries(self) -> int:
        return self._get_value_verbose("negative_sample_max_retries")

    @property
    def no_loss_frequencies(self) -> list:
        return self._as_default_list(self._cfg.get("no_loss_frequencies", []))

    @property
    def num_workers(self) -> int:
        return self._cfg.get("num_workers", 0)

    @property
    def number_of_basins(self) -> int:
        return self._get_value_verbose("number_of_basins")

    @number_of_basins.setter
    def number_of_basins(self, num_basins: int):
        self._cfg["number_of_basins"] = num_basins

    @property
    def ode_method(self) -> str:
        return self._cfg.get("ode_method", "euler")

    @property
    def ode_num_unfolds(self) -> int:
        return self._cfg.get("ode_num_unfolds", 4)

    @property
    def ode_random_freq_lower_bound(self) -> str:
        return self._get_value_verbose("ode_random_freq_lower_bound")

    @property
    def optimizer(self) -> str:
        return self._get_value_verbose("optimizer")

    @property
    def output_activation(self) -> str:
        return self._cfg.get("output_activation", "linear")

    @property
    def output_dropout(self) -> float:
        return self._cfg.get("output_dropout", 0.0)

    @property
    def per_basin_test_periods_file(self) -> Path:
        return self._cfg.get("per_basin_test_periods_file", None)

    @property
    def per_basin_train_periods_file(self) -> Path:
        return self._cfg.get("per_basin_train_periods_file", None)

    @property
    def per_basin_validation_periods_file(self) -> Path:
        return self._cfg.get("per_basin_validation_periods_file", None)

    @property
    def predict_last_n(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("predict_last_n")

    @property
    def random_holdout_from_dynamic_features(self) -> Dict[str, float]:
        return self._as_default_dict(self._cfg.get("random_holdout_from_dynamic_features", {}))

    @property
    def rating_curve_file(self) -> Path:
        return self._get_value_verbose("rating_curve_file")

    @property
    def regularization(self) -> List[Union[str, Tuple[str, float]]]:
        return self._as_default_list(self._cfg.get("regularization", []))

    @property
    def run_dir(self) -> Path:
        return self._cfg.get("run_dir", None)

    @run_dir.setter
    def run_dir(self, folder: Path):
        self._cfg["run_dir"] = folder

    @property
    def save_train_data(self) -> bool:
        return self._cfg.get("save_train_data", False)

    @property
    def save_all_output(self) -> bool:
        return self._cfg.get('save_all_output', False)

    @property
    def save_validation_results(self) -> bool:
        return self._cfg.get("save_validation_results", False)

    @property
    def save_weights_every(self) -> int:
        return self._cfg.get("save_weights_every", 1)

    @property
    def seed(self) -> int:
        return self._cfg.get("seed", None)

    @property
    def transformer_nlayers(self) -> int:
        return self._get_value_verbose("transformer_nlayers")

    @property
    def transformer_positional_encoding_type(self) -> str:
        return self._get_value_verbose("transformer_positional_encoding_type")

    @property
    def transformer_dim_feedforward(self) -> int:
        return self._get_value_verbose("transformer_dim_feedforward")

    @property
    def transformer_positional_dropout(self) -> float:
        return self._get_value_verbose("transformer_positional_dropout")

    @property
    def transformer_dropout(self) -> float:
        return self._get_value_verbose("transformer_dropout")

    @property
    def transformer_nheads(self) -> int:
        return self._get_value_verbose("transformer_nheads")

    @seed.setter
    def seed(self, seed: int):
        if self._cfg.get("seed", None) is None:
            self._cfg["seed"] = seed
        else:
            raise RuntimeError("Seed was already specified and can't be replaced")

    @property
    def seq_length(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("seq_length")

    @property
    def shared_mtslstm(self) -> bool:
        return self._cfg.get("shared_mtslstm", False)

    @property
    def static_attributes(self) -> List[str]:
        if "static_attributes" in self._cfg.keys():
            return self._as_default_list(self._cfg["static_attributes"])
        elif "camels_attributes" in self._cfg.keys():
            warnings.warn("'camels_attributes' устарело. В будущем, используйте 'static_attributes'.",
                          FutureWarning)
            return self._as_default_list(self._cfg["camels_attributes"])
        else:
            return []

    @property
    def statics_embedding(self) -> dict:
        embedding_spec = self._cfg.get("statics_embedding", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def target_loss_weights(self) -> List[float]:
        return self._cfg.get("target_loss_weights", None)

    @property
    def target_noise_std(self) -> float:
        if (self._cfg.get("target_noise_std", None) is None) or (self._cfg["target_noise_std"] == 0):
            return None
        else:
            return self._cfg["target_noise_std"]

    @property
    def target_variables(self) -> List[str]:
        if "target_variables" in self._cfg.keys():
            return self._cfg["target_variables"]
        elif "target_variable" in self._cfg.keys():
            warnings.warn("'target_variable' устарело. В будущем, используйте 'target_variables'", FutureWarning)
            return self._cfg["target_variable"]
        else:
            raise ValueError("Не определены целевые переменные ('target_variables') в конфигурационном файле.")

    @property
    def tau_down(self) -> float:
        return self._get_value_verbose("tau_down")

    @property
    def tau_up(self) -> float:
        return self._get_value_verbose("tau_up")

    @property
    def test_basin_file(self) -> Path:
        return self._get_value_verbose("test_basin_file")

    @property
    def test_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("test_end_date")

    @property
    def test_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("test_start_date")

    @property
    def timestep_counter(self) -> bool:
        return self._cfg.get("timestep_counter", False)

    @property
    def train_basin_file(self) -> Path:
        return self._get_value_verbose("train_basin_file")

    @property
    def train_data_file(self) -> Path:
        return self._cfg.get("train_data_file", None)

    @property
    def train_dir(self) -> Path:
        return self._cfg.get("train_dir", None)

    @train_dir.setter
    def train_dir(self, folder: Path):
        self._cfg["train_dir"] = folder

    @property
    def train_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("train_end_date")

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("train_start_date")

    @property
    def transfer_mtslstm_states(self) -> Dict[str, str]:
        return self._cfg.get("transfer_mtslstm_states", {'h': 'linear', 'c': 'linear'})

    @property
    def umal_extend_batch(self) -> bool:
        return self._cfg.get("umal_extend_batch", False)

    @property
    def use_basin_id_encoding(self) -> bool:
        return self._cfg.get("use_basin_id_encoding", False)

    @property
    def use_frequencies(self) -> List[str]:
        return self._as_default_list(self._cfg.get("use_frequencies", []))

    @property
    def validate_every(self) -> int:
        if (self._cfg.get("validate_every", None) is None) or (self._cfg["validate_every"] < 1):
            return None
        else:
            return self._cfg["validate_every"]

    @property
    def validate_n_random_basins(self) -> int:
        if (self._cfg.get("validate_n_random_basins", None) is None) or (self._cfg["validate_n_random_basins"] < 1):
            return 0
        else:
            return self._cfg["validate_n_random_basins"]

    @validate_n_random_basins.setter
    def validate_n_random_basins(self, n_basins: int):
        self._cfg["validate_n_random_basins"] = n_basins

    @property
    def validation_basin_file(self) -> Path:
        return self._get_value_verbose("validation_basin_file")

    @property
    def validation_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("validation_end_date")

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("validation_start_date")

    @property
    def verbose(self) -> int:
        """"Определяет уровень детализации сообщений.

        0: Регистрирует информационные сообщения, не отображая прогресс-бары
        1: Регистрирует информационные сообщения и прогресс-бары

        Возвращает
        -------
        int
            Уровень детализации.
        """
        return self._cfg.get("verbose", 1)
    
    @property
    def early_stopping(self) -> bool:
        """Использовать ли early stopping. По умолчанию False."""
        early_stopping = self._cfg.get("early_stopping", False)
        if early_stopping and self.validate_every != 1:
            raise ValueError(
                "Early stopping возможна только при проверки после каждой эпохи (validate_every=1). "
                "Установите validate_every=1 в конфигурации для использования early stopping."
            )
        return early_stopping

    @property
    def patience_early_stopping(self) -> int:
        """Количесвто эпох без улучшения до остановки."""
        if self.early_stopping:
            return self._get_value_verbose("patience_early_stopping")
    
    @property
    def attention_heads(self) -> int:
        return self._get_value_verbose("attention_heads")

    @property
    def minimum_epochs_before_early_stopping(self) -> int:
        """Минимальное количество эпох до срабатывания early stopping."""
        if self.early_stopping:
            return self._get_value_verbose("minimum_epochs_before_early_stopping")
    
    @property
    def dynamic_learning_rate(self) -> bool:
        """Использовать ли динамическую скорость обучения. По умолчанию False."""
        return self._cfg.get("dynamic_learning_rate", False)
    
    @property
    def kan_grid_size(self) -> int:
        return self._get_value_verbose("kan_grid_size")
        
    @property
    def kan_hidden_features(self) -> int:
        return self._get_value_verbose("kan_hidden_features")
        
    @property
    def kan_spline_order(self) -> int:
        return self._get_value_verbose("kan_spline_order")

    @property
    def precipitation_variable(self) -> str:
        return self._get_value_verbose("precipitation_variable")

    @property
    def temperature_variable(self) -> str:
        return self._get_value_verbose("temperature_variable")

    @property
    def precip_feature_idx(self) -> int:
        """Индекс precipitation в dynamic_inputs (для RCPIKLA). По умолчанию 0."""
        return self._cfg.get("precip_feature_idx", 0)

    @property
    def temp_feature_idx(self) -> int:
        """Индекс temperature_mean в dynamic_inputs (для RCPIKLA). По умолчанию 1."""
        return self._cfg.get("temp_feature_idx", 1)

    @property
    def phys_loss_alpha(self) -> float:
        """Вес MSE-компоненты в MaskedPhysicsInformedLoss (Zhou & Liu 2025, Eq. 19). По умолчанию 1.0."""
        return self._cfg.get("phys_loss_alpha", 1.0)

    @property
    def phys_loss_beta(self) -> float:
        """Вес физической компоненты L_phys в MaskedPhysicsInformedLoss (Eq. 19). По умолчанию 0.1."""
        return self._cfg.get("phys_loss_beta", 0.1)

    @property
    def summer_loss_gamma(self) -> float:
        """Доп. вес ошибок для летних месяцев (июль–сент.) в SummerWeightedPhysicsLoss. По умолчанию 2.0."""
        return self._cfg.get("summer_loss_gamma", 2.0)

    @property
    def july_loss_weight(self) -> float:
        """Вес первого шага прогноза (июль) в JulyWeightedPhysicsLoss. По умолчанию 4.0."""
        return self._cfg.get("july_loss_weight", 4.0)

    @property
    def patience_dynamic_learning_rate(self) -> int:
        """Количество эпох без улучшения перед уменьшением скорости обучения."""
        if self.dynamic_learning_rate:
            return self._get_value_verbose("patience_dynamic_learning_rate")
        return self._cfg.get("patience_dynamic_learning_rate", 10)

    @property
    def factor_dynamic_learning_rate(self) -> float:
        """Коэффициент уменьшения скорости обучения."""
        if self.dynamic_learning_rate:
            return self._get_value_verbose("factor_dynamic_learning_rate")
        return self._cfg.get("factor_dynamic_learning_rate", 0.5)
    
    def _get_embedding_spec(self, embedding_spec: dict) -> dict:
        if isinstance(embedding_spec, bool) and embedding_spec:  #
            msg = [
                "Семантика аргумента 'dynamics/statics_embedding'  изменилась, и ассоциированные аргументы "
                "'embedding_hiddens/activation/dropout' устарели. Старые спецификации могут перестать поддерживаться "
                "в будущих версиях. Вместо этого, определяйте вложения в виде словаря в аргументе dynamics/statics_embedding."
            ]
            warnings.warn(" ".join(msg), FutureWarning)
            return {
                'type': 'fc',
                'hiddens': self._as_default_list(self._cfg.get("embedding_hiddens", [])),
                'activation': self._cfg.get("embedding_activation", "tanh"),
                'dropout': self._cfg.get("embedding_dropout", 0.0)
            }

        return {
            'type': embedding_spec.get('type', 'fc'),
            'hiddens': self._as_default_list(embedding_spec.get('hiddens', [])),
            'activation': embedding_spec.get('activation', 'tanh'),
            'dropout': embedding_spec.get('dropout', 0.0)
        }


def create_random_name():
    adjectives = ('cossack', 'siberian', 'slavic', 'steppic', 'uralian', 'fresh', 'blonde', 'dark', 'strong', 'simple',
                  'short', 'long', 'grey', 'slow', 'green')
    nouns = ('borshch ', 'cosmonaut', 'knout', 'mammoth', 'pogrom', 'rouble', 'sputnik', 'taiga', 'troika', 'tundra',
             'tsar ', 'verst', 'kopeck', 'intelligentsia')

    rng = random.Random(datetime.now().timestamp())  # системное время как случайное начальное число (seed).
    return rng.choice(adjectives) + '-' + rng.choice(nouns)
