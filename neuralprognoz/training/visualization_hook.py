"""
Хук для генерации графиков после каждой валидации во время обучения (Fixed dimensions).
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

def create_validation_plots(results_dict: dict, epoch: int, img_log_dir: Path, basin_id: str = None, metrics_df: pd.DataFrame = None):
    try:
        img_log_dir.mkdir(parents=True, exist_ok=True)
        
        if not results_dict: return
        
        if basin_id is None:
            basin_id = list(results_dict.keys())[0]
        
        freq_key = list(results_dict[basin_id].keys())[0]
        ds = results_dict[basin_id][freq_key]['xr']
        
        # --- FIX: Безопасное извлечение данных ---
        
        # 1. Данные для графика (День 1)
        if 'time_step' in ds.dims:
            ds_day1 = ds.isel(time_step=0)
            n_steps = len(ds['time_step'])
        else:
            ds_day1 = ds
            n_steps = 1
            
        try:
            # Используем to_dataframe для автоматического выравнивания Obs и Sim по дате
            df_plot = ds_day1.to_dataframe().reset_index()
            
            # Ищем колонки
            obs_col = next((c for c in df_plot.columns if 'obs' in c), None)
            sim_col = next((c for c in df_plot.columns if 'sim' in c), None)
            date_col = next((c for c in df_plot.columns if 'date' in c), None)
            
            if obs_col and sim_col and date_col:
                df_plot = df_plot[[date_col, obs_col, sim_col]].rename(
                    columns={obs_col: 'Observed', sim_col: 'Simulated', date_col: 'date'}
                ).dropna()
            else:
                df_plot = pd.DataFrame()
                
        except Exception as e:
            LOGGER.warning(f"Ошибка конвертации в DataFrame на эпохе {epoch}: {e}")
            df_plot = pd.DataFrame()

        # 2. Данные для метрик S/sigma (Все дни)
        # Извлекаем сырые массивы
        obs_raw = ds['streamflow_obs'].values
        sim_raw = ds['streamflow_sim'].values
        
        # FIX: Принудительное сплющивание Obs (если он пришел как [N, 1] или [N, 4])
        # Для корректности берем только первый столбец, если он многомерный, так как obs везде одинаковый
        if obs_raw.ndim > 1:
            obs_raw = obs_raw[:, 0]
        obs_raw = obs_raw.flatten()
            
        # Подготовка Sim
        if n_steps == 1 and sim_raw.ndim == 1:
            sim_raw = sim_raw.reshape(-1, 1)

        # Считаем Sigma
        obs_clean = obs_raw[~np.isnan(obs_raw)]
        if len(obs_clean) > 1:
            sigma_delta = np.std(np.diff(obs_clean), ddof=1)
        else:
            sigma_delta = np.nan

        skill_stats = []
        for i in range(n_steps):
            if sim_raw.ndim > 1:
                sim_step = sim_raw[:, i]
            else:
                sim_step = sim_raw
            
            # Обрезка по длине
            min_len = min(len(obs_raw), len(sim_step))
            obs_v = obs_raw[:min_len]
            sim_v = sim_step[:min_len]
            
            mask = ~(np.isnan(obs_v) | np.isnan(sim_v))
            if np.sum(mask) > 10:
                rmse = np.sqrt(np.mean((obs_v[mask] - sim_v[mask])**2))
                val = rmse / sigma_delta if sigma_delta > 0 else 0
                skill_stats.append(val)
            else:
                skill_stats.append(0)

        # --- РИСОВАНИЕ ---
        if df_plot.empty and not skill_stats:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # График 1
        if not df_plot.empty:
            axes[0].plot(df_plot['date'], df_plot['Observed'], 'k-', lw=1, label='Obs')
            axes[0].plot(df_plot['date'], df_plot['Simulated'], 'r-', lw=1, label='Sim')
            axes[0].set_title(f"Epoch {epoch} (Day 1)")
            axes[0].legend()
            fig.autofmt_xdate()
        
        # График 2
        if skill_stats:
            axes[1].bar(range(1, len(skill_stats)+1), skill_stats)
            axes[1].axhline(0.8, color='g', ls='--')
            axes[1].set_title("S/Sigma by Day")
            
        plt.tight_layout()
        plt.savefig(img_log_dir / f'epoch_{epoch:03d}.png', dpi=100)
        plt.close()
        
    except Exception as e:
        LOGGER.warning(f"Не удалось создать график для эпохи {epoch}: {e}")