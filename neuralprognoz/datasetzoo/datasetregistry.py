from typing import Type

from neuralprognoz.datasetzoo.basedataset import BaseDataset
from neuralprognoz.utils.config import Config


class DatasetRegistry:
    """Класс, который регистрирует классы наборов данных, которые можно использовать с neuralprognoz."""

    def __init__(self):
        self.__dataset_class = {}

    def register_dataset_class(self, key: str, new_class: Type):
        """Добавляет новый класс набора данных в реестр наборов данных.

        Параметры
        ----------
        key : str
            Уникальный идентификатор для класса набора данных. Этот ключ будет использоваться в файлах конфигурации
            для указания, какой набор данных использовать.
        new_class : Type
            Класс набора данных для регистрации. Должен быть подклассом BaseDataset.

        Возвращает
        -------
        None

        Вызывает
        ------
        TypeError
            Если предоставленный класс не является подклассом BaseDataset.

        Примеры
        --------
        >>> registry = DatasetZooRegistry()
        >>> registry.register_dataset_class("my_dataset", MyCustomDataset)
        """
        if not issubclass(new_class, BaseDataset):
            raise TypeError(f"Класс {type(new_class)} не является подклассом BaseDataset.")
        self.__dataset_class[key] = new_class

    def instantiate_dataset(self,
                            cfg: Config,
                            is_train: bool,
                            period: str,
                            basin: str = None,
                            additional_features: list = [],
                            id_to_int: dict = {},
                            scaler: dict = {}) -> BaseDataset:
        """Создаёт и возвращает экземпляр класса набора данных на основе конфигурации.

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
            Экземпляр соответствующего класса набора данных на основе конфигурации.

        Вызывает
        ------
        NotImplementedError
            Если для набора данных, указанного в конфигурации, не реализован класс набора данных.
        """
        dataset_key = cfg.dataset.lower()
        Dataset = self.__dataset_class.get(dataset_key, None)
        if Dataset is None:
            raise NotImplementedError(f"Для набора данных {cfg.dataset} не реализован класс набора данных")

        return Dataset(cfg=cfg,
                       is_train=is_train,
                       period=period,
                       basin=basin,
                       additional_features=additional_features,
                       id_to_int=id_to_int,
                       scaler=scaler)
