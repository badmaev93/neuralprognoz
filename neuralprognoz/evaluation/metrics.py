
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats, signal
from xarray.core.dataarray import DataArray

from neuralprognoz.datautils import utils
from neuralprognoz.utils.errors import AllNaNError

LOGGER = logging.getLogger(__name__)


def get_available_metrics() -> List[str]:
    return [
        "NSE", "MSE", "RMSE", "KGE", "Alpha-NSE", "Pearson-r", "Beta-KGE", "Beta-NSE", "FHV", "FMS", "FLV",
        "Peak-Timing", "Missed-Peaks", "Peak-MAPE", "S/sigma_delta"
    ]


def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError(f"Размеры наблюдений {obs.shape} и симуляций {sim.shape} должны совпадать")

def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
    idx = (~sim.isnull()) & (~obs.isnull())
    obs = obs[idx]
    sim = sim[idx]
    return obs, sim

def _check_all_nan(obs: DataArray, sim: DataArray):
    if all(obs.isnull()):
        raise AllNaNError("Все наблюдаемые значения - NaN")
    if all(sim.isnull()):
        raise AllNaNError("Все симулированные значения - NaN")

def nse(obs: DataArray, sim: DataArray) -> float:
    _validate_inputs(obs, sim)
    obs, sim = _mask_valid(obs, sim)
    if len(obs) < 2: return np.nan
    denominator = ((obs - obs.mean())**2).sum()
    numerator = ((sim - obs)**2).sum()
    return float(1 - numerator / denominator)

def kge(obs: DataArray, sim: DataArray) -> float:
    _validate_inputs(obs, sim)
    obs, sim = _mask_valid(obs, sim)
    if len(obs) < 2: return np.nan
    r, _ = stats.pearsonr(obs.values, sim.values)
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

def rmse(obs: DataArray, sim: DataArray) -> float:
    _validate_inputs(obs, sim)
    obs, sim = _mask_valid(obs, sim)
    return float(np.sqrt(((sim - obs)**2).mean()))

def s_sigma_delta(obs: DataArray, sim: DataArray) -> float:
    """Рассчет S/sigma_delta по русской гидрологической методологии.
    Значения <= 0.8 указывают на эффективное прогнозирование.
    """
    obs_vals = obs.values.flatten() if hasattr(obs, 'values') else np.array(obs).flatten()
    sim_vals = sim.values.flatten() if hasattr(sim, 'values') else np.array(sim).flatten()
    
    mask = ~(np.isnan(obs_vals) | np.isnan(sim_vals))
    obs_vals = obs_vals[mask]
    sim_vals = sim_vals[mask]
    
    if len(obs_vals) < 2:
        return np.nan
    
    s_model = np.sqrt(np.mean((obs_vals - sim_vals)**2))
    
    delta_changes = np.diff(obs_vals)
    
    if len(delta_changes) == 0:
        return np.nan
    delta_changes = np.diff(obs_vals)
    
    if len(delta_changes) == 0:
        return np.nan
        
    sigma_delta = np.std(delta_changes, ddof=1)
    
    return float(s_model / sigma_delta) if sigma_delta > 0 else np.nan


def calculate_metrics(obs: DataArray, sim: DataArray, metrics: List[str], resolution: str="1D", datetime_coord: str=None) -> Dict[str, float]:
    _check_all_nan(obs, sim)
    values = {}
    
    for metric in metrics:
        m_lower = metric.lower()
        try:
            if m_lower == "nse":
                values["NSE"] = nse(obs, sim)
            elif m_lower == "kge":
                values["KGE"] = kge(obs, sim)
            elif m_lower == "rmse":
                values["RMSE"] = rmse(obs, sim)
            elif m_lower in ["s/σδ", "s/sigma_delta", "s_sigma_delta"]:
                values["S/sigma_delta"] = s_sigma_delta(obs, sim)
        except Exception as e:
            LOGGER.warning(f"Error calculating {metric}: {e}")
            values[metric] = np.nan
            
    return values
