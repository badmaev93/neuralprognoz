from typing import Type

from neuralprognoz.datasetzoo.basedataset import BaseDataset
from neuralprognoz.datasetzoo.genericdataset import GenericDataset
from neuralprognoz.utils.config import Config
from neuralprognoz.datasetzoo.datasetregistry import DatasetRegistry


def get_dataset(cfg: Config,
                is_train: bool,
                period: str,
                basin: str = None,
                additional_features: list = [],
                id_to_int: dict = {},
                scaler: dict = {}) -> BaseDataset:
    """Получить экземпляр набора данных в зависимости от конфигурации запуска.

    Класс набора данных 'generic' может использоваться для любого типа набора данных, если
    он находится в правильном формате.

    Новые классы наборов данных могут быть добавлены в начале времени выполнения с помощью функции register_dataset().

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

    Возвращает
    -------
    BaseDataset
        Новый экземпляр набора данных в зависимости от конфигурации запуска.

    Вызывает
    ------
    NotImplementedError
        Если для аргумента 'dataset' в конфиге не реализован класс набора данных.
    """
    global _datasetZooRegistry

    return _datasetZooRegistry.instantiate_dataset(cfg, is_train, period, basin, additional_features, id_to_int, scaler)


def register_dataset(key: str, new_class: Type):
    """Добавляет класс набора данных в реестр наборов данных.
    
    Этот класс должен наследоваться от BaseDataset. Новый класс набора данных должен быть добавлен в начале времени выполнения.

    Параметры
    ----------
    key : str
        Ключ набора данных, который устанавливается в файле конфигурации.

    new_class : Type
        Новый класс Dataset для регистрации.

    Возвращает
    -------
    None

    Вызывает
    ------
    TypeError
        Если новый класс не наследуется от BaseDataset.
    """
    global _datasetZooRegistry
    _datasetZooRegistry.register_dataset_class(key, new_class)


_datasetZooRegistry: DatasetRegistry = DatasetRegistry()

_datasetZooRegistry.register_dataset_class("generic", GenericDataset)
