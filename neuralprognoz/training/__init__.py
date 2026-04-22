import logging
import warnings
from typing import List

import torch

import neuralprognoz.training.loss as loss
from neuralprognoz.training import regularization
from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Получает конкретный объект оптимизатора, в зависимости от конфигурации запуска.
    
    В настоящее время поддерживаются только 'Adam' и 'AdamW'.
    """
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate[0])
    else:
        raise NotImplementedError(f"{cfg.optimizer} не реализован или не связан в `get_optimizer()`")

    return optimizer


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Получает объект функции потерь, в зависимости от конфигурации запуска.
    """
    # Приводим к нижнему регистру для надежности
    loss_name = cfg.loss.lower()

    if loss_name == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif loss_name == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif loss_name == "weightednse":
        warnings.warn("'WeightedNSE loss' был удален. Используйте 'NSE' с 'target_loss_weights'", FutureWarning)
        loss_obj = loss.MaskedNSELoss(cfg)
    elif loss_name == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif loss_name == "ssigmadeltaloss" or loss_name == "maskedssigmadeltaloss":
        loss_obj = loss.MaskedSSigmaDeltaLoss(cfg)
    elif loss_name == "hybriddiffloss" or loss_name == "maskedhybriddiffloss":
        loss_obj = loss.MaskedHybridDiffLoss(cfg)
    elif loss_name in ("physicsinformedloss", "maskedphysicsinformedloss", "physicsloss"):
        loss_obj = loss.MaskedPhysicsInformedLoss(cfg)
    elif loss_name in ("summerweightedphysicsloss",):
        loss_obj = loss.SummerWeightedPhysicsLoss(cfg)
    elif loss_name in ("julyweightedphysicsloss",):
        loss_obj = loss.JulyWeightedPhysicsLoss(cfg)
    elif loss_name == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif loss_name == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif loss_name == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} не реализован или не связан в `get_loss()`")

    return loss_obj


def get_regularization_obj(cfg: Config) -> List[regularization.BaseRegularization]:
    """Получает список объектов регуляризации.
    """
    regularization_modules = []
    # Если regularization в конфиге None или не список, возвращаем пустой
    if not cfg.regularization:
        return regularization_modules

    for reg_item in cfg.regularization:
        if isinstance(reg_item, str):
            reg_name = reg_item
            reg_weight = 1.0
        else:
            reg_name, reg_weight = reg_item
            
        if reg_name == "tie_frequencies":
            regularization_modules.append(regularization.TiedFrequencyMSERegularization(cfg=cfg, weight=reg_weight))
        elif reg_name == "forecast_overlap":
            regularization_modules.append(regularization.ForecastOverlapMSERegularization(cfg=cfg, weight=reg_weight))
        elif reg_name == "snowmelt_consistency":
            regularization_modules.append(regularization.SnowmeltConsistencyRegularization(cfg=cfg, weight=reg_weight))
        else:
            raise NotImplementedError(f"{reg_name} не реализован или не связан в `get_regularization_obj()`.")

    return regularization_modules
