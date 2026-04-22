import warnings

import torch.nn as nn

from neuralprognoz.modelzoo.arlstm import ARLSTM
from neuralprognoz.modelzoo.cudalstm import CudaLSTM
from neuralprognoz.modelzoo.mamba import Mamba
from neuralprognoz.modelzoo.customlstm import CustomLSTM
from neuralprognoz.modelzoo.ealstm import EALSTM
from neuralprognoz.modelzoo.embcudalstm import EmbCudaLSTM
from neuralprognoz.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from neuralprognoz.modelzoo.hybridmodel import HybridModel
from neuralprognoz.modelzoo.gru import GRU
from neuralprognoz.modelzoo.mclstm import MCLSTM
from neuralprognoz.modelzoo.mtslstm import MTSLSTM
from neuralprognoz.modelzoo.multihead_forecast_lstm import MultiHeadForecastLSTM
from neuralprognoz.modelzoo.odelstm import ODELSTM
from neuralprognoz.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from neuralprognoz.modelzoo.stacked_forecast_lstm import StackedForecastLSTM
from neuralprognoz.modelzoo.transformer import Transformer
from neuralprognoz.utils.config import Config
from neuralprognoz.modelzoo.pikla import PIKLA
from neuralprognoz.modelzoo.rcpikla import RCPIKLA

SINGLE_FREQ_MODELS = [
    "cudalstm",
    "ealstm", 
    "customlstm", 
    "embcudalstm", 
    "gru", 
    "transformer",
    "mamba",
    "mclstm", 
    "arlstm",
    "handoff_forecast_lstm",
    "sequential_forecast_lstm",
    "multihead_forecast_lstm",
    "stacked_forecast_lstm"
]
AUTOREGRESSIVE_MODELS = ['arlstm']


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.
    
    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Возвращает
    -------
    nn.Module
        Новый экземпляр модели, указанный в конфигурации.
    """
    if cfg.model.lower() in SINGLE_FREQ_MODELS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"Модель {cfg.model} не поддерживает множественные частоты.")

    if cfg.model.lower() not in AUTOREGRESSIVE_MODELS and cfg.autoregressive_inputs:
        raise ValueError(f"Модель {cfg.model} не поддерживает авторегрессию.")

    if cfg.model.lower() != "mclstm" and cfg.mass_inputs:
        raise ValueError(f"Использование 'mass_inputs' с моделью {cfg.model} не поддерживается.")

    if cfg.model.lower() == "arlstm":
        model = ARLSTM(cfg=cfg)
    elif cfg.model.lower() == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "ealstm":
        model = EALSTM(cfg=cfg)
    elif cfg.model.lower() == "customlstm":
        model = CustomLSTM(cfg=cfg)
    elif cfg.model.lower() == "lstm":
        warnings.warn(
            "Класс `LSTM` был переименован в `CustomLSTM`. Поддержка `LSTM` будет прекращена в будущем.",
            FutureWarning)
        model = CustomLSTM(cfg=cfg)
    elif cfg.model.lower() == "gru":
        model = GRU(cfg=cfg)
    elif cfg.model.lower() == "embcudalstm":
        model = EmbCudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "mtslstm":
        model = MTSLSTM(cfg=cfg)
    elif cfg.model.lower() == "odelstm":
        model = ODELSTM(cfg=cfg)
    elif cfg.model.lower() == "mclstm":
        model = MCLSTM(cfg=cfg)
    elif cfg.model.lower() == "transformer":
        model = Transformer(cfg=cfg)
    elif cfg.model.lower() == "mamba":
        model = Mamba(cfg=cfg)
    elif cfg.model.lower() == "handoff_forecast_lstm":
        model = HandoffForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "multihead_forecast_lstm":
        model = MultiHeadForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "sequential_forecast_lstm":
        model = SequentialForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "stacked_forecast_lstm":
        model = StackedForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "hybrid_model":
        model = HybridModel(cfg=cfg)
    elif cfg.model.lower() == "pikla":
        model = PIKLA(cfg=cfg)
    elif cfg.model.lower() == "rcpikla":
        model = RCPIKLA(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} не реализована или не связана в `get_model()`")

    return model
