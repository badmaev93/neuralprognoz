#!/usr/bin/env python
# Отключ colorama полностью для Windows перед любыми импортами
from neuralprognoz.evaluation.metrics import nse, rmse
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import pickle
from neuralprognoz.utils.logging_utils import setup_logging
from neuralprognoz.utils.config import Config
from neuralprognoz.training.train import start_training
from neuralprognoz.evaluation.evaluate import start_evaluation
import os
import sys
if sys.platform == 'win32':
    os.environ['NO_COLOR'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.modules['colorama'] = None
    sys.modules['colorama.ansitowin32'] = None
    sys.modules['colorama.win32'] = None

import argparse
import logging
from pathlib import Path

# убедит, что директория с кодом находится в пути, даже если пакет не
# установлен через setup.py
sys.path.append(str(Path(__file__).parent.parent))
matplotlib.use('Agg')  #  backend без GUI


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        choices=[
            "train",
            "continue_training",
            "finetune",
            "evaluate"])
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--run-dir', type=str)
    parser.add_argument(
        '--epoch',
        type=int,
        help="Эпоха, для которой должна быть проведена оценка модели")
    parser.add_argument(
        '--period',
        type=str,
        choices=[
            "train",
            "validation",
            "test"],
        default="test")
    parser.add_argument(
        '--gpu',
        type=int,
        help="ID GPU для использования. Переопределяет аргумент 'device' в конфиге. Значение < 0 — CPU.")
    args = vars(parser.parse_args())

    if (args["mode"] in ["train", "finetune"]) and (
            args["config_file"] is None):
        raise ValueError("Не указан путь к конфигурационному файлу")

    if (args["mode"] == "continue_training") and (args["run_dir"] is None):
        raise ValueError("Не указан путь к директории запуска")

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Не указан путь к директории запуска")

    return args


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Рассчитывает процентное смещение (Percent Bias)."""
    sum_obs = np.sum(obs)
    if np.isclose(sum_obs, 0):
        return np.nan
    return 100 * np.sum(sim - obs) / sum_obs


def calculate_overall_s_sigma_delta(df: pd.DataFrame) -> float:
    """
    Рассчитывает общий S/σΔ для всего периода.

    Возвращает
    -------
    float
        Значение S/σΔ для всего периода (чем меньше, тем лучше)
    """
    if len(df) < 2:
        return np.nan

    obs = df['Observed'].values
    sim = df['Simulated'].values

    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[valid_mask]
    sim = sim[valid_mask]

    if len(obs) < 2:
        return np.nan

    delta_changes = np.diff(obs)

    if len(delta_changes) == 0:
        return np.nan

    delta_mean = np.mean(delta_changes)
    sigma_delta = np.sqrt(
        np.sum((delta_changes - delta_mean)**2) / (len(delta_changes) - 1))

    s_lstm = np.sqrt(np.mean((obs - sim)**2))

    return s_lstm / sigma_delta if sigma_delta > 0 else np.nan


def calculate_s_sigma_delta_operational(
        df: pd.DataFrame,
        max_lead_time: int = 15) -> pd.DataFrame:
    """
    Рассчитывает S/sigma_delta для оперативного режима прогнозирования.

    В оперативном режиме каждые 15 дней модель получает фактические данные
    и делает новый прогноз на следующие 15 дней.
    """
    skill_data = []

    if 'lead_time' in df.columns:
        for lead_time in range(1, max_lead_time + 1):
            lead_data = df[df['lead_time'] == lead_time].copy()

            if len(lead_data) < 2:
                continue

            obs_values = lead_data['observed'].values
            sim_values = lead_data['predicted'].values

            valid_mask = ~(np.isnan(obs_values) | np.isnan(sim_values))
            obs_values = obs_values[valid_mask]
            sim_values = sim_values[valid_mask]

            if len(obs_values) < 2:
                continue

            delta_changes = np.diff(obs_values)
            if len(delta_changes) > 1:
                delta_mean = np.mean(delta_changes)
                sigma_delta = np.sqrt(
                    np.sum((delta_changes - delta_mean)**2) / (len(delta_changes) - 1))
            else:
                sigma_delta = np.std(delta_changes) if len(
                    delta_changes) > 0 else np.nan

            s_model = np.sqrt(np.mean((obs_values - sim_values)**2))

            skill_ratio = s_model / sigma_delta if sigma_delta > 0 else np.nan

            skill_data.append({
                'Заблаговременность, сут.': f"{lead_time} (n={len(obs_values)})",
                'S/σΔ (LSTM)': skill_ratio,
                'n_forecasts': len(obs_values)
            })

    return pd.DataFrame(skill_data).set_index('Заблаговременность, сут.')


def calculate_s_sigma_delta(df: pd.DataFrame,
                            max_lead_time: int = 15) -> pd.DataFrame:
    """
    Рассчитывает S/sigma_delta для разной заблаговременности согласно гидрологической методике.

    Методика: после каждых 15 суток заблаговременность начинается заново (1, 2, 3... суток).
    σΔ рассчитывается как среднеквадратическое отклонение изменений для каждой заблаговременности.
    """
    skill_data = []
    df_date_indexed = df.set_index('date')

    n_total = len(df_date_indexed)
    period_length = 15
    n_full_periods = n_total // period_length
    remainder_days = n_total % period_length

    print(
        f"Всего дней: {n_total}, полных периодов: {n_full_periods}, остаток: {remainder_days}")

    lstm_errors_by_lead = {dt: [] for dt in range(1, max_lead_time + 1)}
    inertial_errors_by_lead = {dt: [] for dt in range(1, max_lead_time + 1)}
    delta_changes_by_lead = {dt: [] for dt in range(1, max_lead_time + 1)}

    for period in range(n_full_periods):
        start_idx = period * period_length
        end_idx = start_idx + period_length

        period_data = df_date_indexed.iloc[start_idx:end_idx]

        for dt in range(1, min(max_lead_time + 1, period_length)):
            if dt >= len(period_data):
                continue

            obs_values = period_data['Observed'].iloc[dt:].values

            inertial_values = period_data['Observed'].iloc[:-dt].values

            lstm_values = period_data['Simulated'].iloc[:-dt].values

            if len(obs_values) == 0 or len(inertial_values) == 0 or len(
                    obs_values) != len(lstm_values):
                continue

            if len(obs_values) > 1:
                delta_changes = np.diff(obs_values)
                delta_changes_by_lead[dt].extend(delta_changes)

            lstm_squared_errors = (obs_values - lstm_values)**2
            inertial_squared_errors = (obs_values - inertial_values)**2

            lstm_errors_by_lead[dt].extend(lstm_squared_errors)
            inertial_errors_by_lead[dt].extend(inertial_squared_errors)

    if remainder_days > 0:
        start_idx = n_full_periods * period_length
        remainder_data = df_date_indexed.iloc[start_idx:]

        for dt in range(1, min(max_lead_time + 1, remainder_days)):
            if dt >= len(remainder_data):
                continue

            obs_values = remainder_data['Observed'].iloc[dt:].values

            inertial_values = remainder_data['Observed'].iloc[:-dt].values

            lstm_values = remainder_data['Simulated'].iloc[:-dt].values

            if len(obs_values) == 0 or len(inertial_values) == 0 or len(
                    obs_values) != len(lstm_values):
                continue

            if len(obs_values) > 1:
                delta_changes = np.diff(obs_values)
                delta_changes_by_lead[dt].extend(delta_changes)

            lstm_squared_errors = (obs_values - lstm_values)**2
            inertial_squared_errors = (obs_values - inertial_values)**2

            lstm_errors_by_lead[dt].extend(lstm_squared_errors)
            inertial_errors_by_lead[dt].extend(inertial_squared_errors)

    for dt in range(1, max_lead_time + 1):
        if not delta_changes_by_lead[dt] or not lstm_errors_by_lead[dt]:
            continue

        delta_changes = np.array(delta_changes_by_lead[dt])
        if len(delta_changes) > 1:
            delta_mean = np.mean(delta_changes)
            sigma_delta = np.sqrt(
                np.sum((delta_changes - delta_mean)**2) / (len(delta_changes) - 1))
        else:
            sigma_delta = np.std(delta_changes) if len(
                delta_changes) > 0 else np.nan

        lstm_squared_errors = np.array(lstm_errors_by_lead[dt])
        inertial_squared_errors = np.array(inertial_errors_by_lead[dt])

        s_lstm = np.sqrt(np.mean(lstm_squared_errors))
        s_inertial = np.sqrt(np.mean(inertial_squared_errors))

        skill_lstm = s_lstm / sigma_delta if sigma_delta > 0 else np.nan
        skill_inertial = s_inertial / sigma_delta if sigma_delta > 0 else np.nan

        n_periods = n_full_periods + (1 if dt <= remainder_days else 0)

        skill_data.append({
            'Заблаговременность, сут.': f"{dt} (n={n_periods})",
            'S/σΔ (LSTM)': skill_lstm,
            'S/σΔ (Inertial)': skill_inertial
        })

    return pd.DataFrame(skill_data).set_index('Заблаговременность, сут.')


def create_evaluation_plots(run_dir: Path):
    """Создает, сохраняет дашборд с результатами и экспортирует данные в Excel."""
    logging.info("Создание расширенного отчета по результатам оценки...")

    try:
        test_dir = run_dir / 'test'
        if not test_dir.exists():
            test_dir = run_dir / 'validation'
            if not test_dir.exists():
                logging.error(
                    f"Папка с результатами (test/validation) не найдена в {run_dir}")
                return

        results_path = sorted(list(test_dir.glob('**/results.p')))
        if not results_path:
            results_path = sorted(list(test_dir.glob('**/test_results.p')))

        if not results_path:
            logging.error(f"Файл results.p не найден в {test_dir}")
            return

        results_path = results_path[-1]
        metrics_path = results_path.parent / 'metrics.csv'
        if not metrics_path.exists():
            metrics_path = results_path.parent / 'test_metrics.csv'

        logging.info(f"Загружаем результаты из: {results_path}")

        with open(results_path, 'rb') as f:
            logging.warning("SECURITY WARNING: Loading a pickle file can execute arbitrary code. "
                            "Ensure that the results file comes from a trusted source.")
            results_data = pickle.load(f)

        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path, index_col='basin')
        else:
            logging.warning(
                "Файл метрик не найден, таблица метрик будет пустой.")
            metrics_df = pd.DataFrame()

        if not results_data:
            logging.error("Результаты пусты")
            return

        basin_id = list(results_data.keys())[0]
        logging.info(f"Обрабатываем данные для бассейна: {basin_id}")

        freq_key = list(results_data[basin_id].keys())[0]
        if 'xr' not in results_data[basin_id][freq_key]:
            logging.error("Нет xarray данных в результатах")
            return

        ds = results_data[basin_id][freq_key]['xr']

        is_multistep = 'time_step' in ds.dims

        if is_multistep:
            ds_plot = ds.isel(time_step=0)
            logging.info(
                f"Обнаружен многодневный прогноз. Для графика берем time_step=0.")
        else:
            ds_plot = ds

        df_viz = ds_plot.to_dataframe()
        df_viz = df_viz.rename(
            columns={
                'streamflow_obs': 'Observed',
                'streamflow_sim': 'Simulated'})
        df_viz = df_viz.reset_index().rename(columns={'date': 'date'})
        df_viz = df_viz[['date', 'Observed', 'Simulated']].dropna()

        if df_viz.empty:
            logging.warning("Нет данных после удаления NaN.")
            return

        # 2. Расчет S/sigma_delta для КАЖДОЙ заблаговременности
        skill_data = []
        if is_multistep:
            obs_all = ds['streamflow_obs'].isel(time_step=0).values
        else:
            obs_all = ds['streamflow_obs'].values
        if obs_all.ndim == 2:
            obs_all = obs_all[:, 0]
        elif obs_all.ndim == 3:
            obs_all = obs_all[:, 0, 0]

        obs_diff = np.diff(obs_all, axis=0)
        obs_diff = obs_diff[~np.isnan(obs_diff)]
        if len(obs_diff) > 1:
            sigma_delta_val = np.std(obs_diff, ddof=1)
        else:
            sigma_delta_val = np.nan

        if is_multistep:
            n_steps = len(ds['time_step'])
            for step in range(n_steps):
                sim_step = ds['streamflow_sim'].isel(time_step=step).values


                valid_mask = ~(np.isnan(obs_all) | np.isnan(sim_step))
                if np.sum(valid_mask) < 2:
                    continue

                mse = np.mean((obs_all[valid_mask] - sim_step[valid_mask])**2)
                s_metric = np.sqrt(mse)

                ratio = s_metric / sigma_delta_val if sigma_delta_val > 0 else np.nan

                skill_data.append({
                    'Заблаговременность': step + 1,
                    'S/σΔ': ratio
                })
        else:
            mse = np.mean((df_viz['Observed'] - df_viz['Simulated'])**2)
            s_metric = np.sqrt(mse)
            ratio = s_metric / sigma_delta_val if sigma_delta_val > 0 else np.nan
            skill_data.append({'Заблаговременность': 1, 'S/σΔ': ratio})

        s_sigma_df = pd.DataFrame(skill_data).set_index('Заблаговременность')

        # --- Создание дашборда ---
        fig, axes = plt.subplots(
            2, 2, figsize=(
                20, 12), gridspec_kw={
                'hspace': 0.35, 'wspace': 0.25})
        fig.suptitle(
            f'Анализ: {basin_id} (Модель на {
                len(skill_data)} дн.)',
            fontsize=20)

        # 1. Гидрограф (День 1)
        ax1 = axes[0, 0]
        ax1.plot(df_viz['date'], df_viz['Observed'], 'k-', label='Наблюдения')
        ax1.plot(
            df_viz['date'],
            df_viz['Simulated'],
            'r--',
            label='Прогноз (1-й день)')
        ax1.set_title('Гидрограф (Заблаговременность: 1 сутки)')
        ax1.set_ylabel('Сток')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        fig.autofmt_xdate()

        # 2. S/sigma по заблаговременности
        ax2 = axes[0, 1]
        if not s_sigma_df.empty:
            s_sigma_df.plot(kind='bar', ax=ax2, color='#4C72B0', legend=False)
            ax2.axhline(
                0.8,
                color='darkgreen',
                linestyle='--',
                label='Порог 0.8')
            ax2.set_title('Качество прогноза по дням (S/σΔ)')
            ax2.set_ylabel('S/σΔ')
            ax2.set_xlabel('День прогноза')
            ax2.grid(axis='y', linestyle='--', alpha=0.6)
            for p in ax2.patches:
                ax2.annotate(f'{p.get_height():.2f}',
                             (p.get_x() + p.get_width() / 2.,
                              p.get_height()),
                             ha='center',
                             va='bottom',
                             fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Нет данных S/σΔ", ha='center')

        # 3. Scatter Plot (День 1)
        ax3 = axes[1, 0]
        ax3.scatter(df_viz['Observed'], df_viz['Simulated'], alpha=0.5, s=10)
        min_val = min(df_viz['Observed'].min(), df_viz['Simulated'].min())
        max_val = max(df_viz['Observed'].max(), df_viz['Simulated'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--')
        ax3.set_title('Диаграмма рассеяния (1-й день)')
        ax3.set_xlabel('Наблюдения')
        ax3.set_ylabel('Прогноз')
        ax3.grid(True)

        # 4. Текст метрик
        ax4 = axes[1, 1]
        ax4.axis('off')
        if not metrics_df.empty and basin_id in metrics_df.index:
            txt = metrics_df.loc[basin_id].round(3).to_string()
        else:
            txt = "Метрики не найдены в CSV"

        ax4.text(0.1, 0.8, "Общие метрики (усредненные):", fontweight='bold')
        ax4.text(0.1, 0.6, txt, family='monospace')

        plot_path = results_path.parent / 'dashboard.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"График сохранен: {plot_path}")

        excel_path = results_path.parent / 'full_analysis.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            df_viz.to_excel(writer, sheet_name="Forecast_Day1", index=False)
            s_sigma_df.to_excel(writer, sheet_name="Skill_Scores")


        logging.info(f"Excel сохранен: {excel_path}")

    except Exception as e:
        logging.error(f"Ошибка в create_evaluation_plots: {e}", exc_info=True)


def _main():
    args = _get_args()
    if (args["run_dir"] is not None) and (args["mode"] == "evaluate"):
        setup_logging(str(Path(args["run_dir"]) / "output.log"))

    if args["mode"] == "train":
        start_run(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["mode"] == "continue_training":
        continue_run(
            run_dir=Path(
                args["run_dir"]),
            config_file=Path(
                args["config_file"]) if args["config_file"] is not None else None,
            gpu=args["gpu"])
    elif args["mode"] == "finetune":
        finetune(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["mode"] == "evaluate":
        eval_run(
            run_dir=Path(
                args["run_dir"]),
            period=args["period"],
            epoch=args["epoch"],
            gpu=args["gpu"])
        create_evaluation_plots(run_dir=Path(args["run_dir"]))
    else:
        raise RuntimeError(f"Неизвестный режим {args['mode']}")


def start_run(config_file: Path, gpu: int = None):
    """
    Запуск обучения модели.

    Параметры
    ----------
    config_file : Path
        Путь к конфигурационному файлу (.yml), определяющему настройки для конкретного запуска.
    gpu : int, необязательно
        ID GPU для использования. Переопределяет параметр 'device' в конфиге. Значение меньше нуля — CPU.
        Не используйте этот аргумент, если хотите использовать устройство, указанное в конфиге, например MPS.
    """

    config = Config(config_file)


    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config)


def continue_run(run_dir: Path, config_file: Path = None, gpu: int = None):
    """
    Продолжить обучение модели.

    Параметры
    ----------
    run_dir : Path
        Путь к директории запуска.
    config_file : Path, необязательно
        Путь к дополнительному конфигу. Каждый параметр из этого файла переопределит исходный конфиг запуска.
    gpu : int, необязательно
        ID GPU для использования. Переопределяет параметр 'device' в конфиге. Значение меньше нуля — CPU.
        Не используйте этот аргумент, если хотите использовать устройство, указанное в конфиге, например MPS.
    """

    base_config = Config(run_dir / "config.yml")

    if config_file is not None:
        base_config.update_config(config_file)
        base_config.run_dir = run_dir

    base_config.is_continue_training = True
    if gpu is not None and gpu >= 0:
        base_config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        base_config.device = "cpu"

    start_training(base_config)


def finetune(config_file: Path = None, gpu: int = None):
    """
    Доводка (finetune) предобученной модели.

    Параметры
    ----------
    config_file : Path, необязательно
        Путь к дополнительному конфигу. Каждый параметр из этого файла переопределит исходный конфиг запуска.
        Конфиг для доводки должен содержать параметр `base_run_dir`, указывающий на папку с предобученной моделью,
        а также 'finetune_modules' для указания, какие части модели будут обучаться при доводке.
    gpu : int, необязательно
        ID GPU для использования. Переопределяет параметр 'device' в конфиге. Значение меньше нуля — CPU.
        Не используйте этот аргумент, если хотите использовать устройство, указанное в конфиге, например MPS.
    """

    temp_config = Config(config_file)
    if not temp_config.finetune_modules:
        raise ValueError(
            "Для доводки необходимо указать хотя бы одну часть модели в 'finetune_modules'.")

    config = Config(temp_config.base_run_dir / "config.yml")
    config.update_config({'run_dir': None, 'experiment_name': None})
    config.update_config(config_file)
    config.is_finetuning = True

    config.is_continue_training = False


    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config)


def eval_run(run_dir: Path, period: str, epoch: int = None, gpu: int = None):
    """
    Запуск оценки обученной модели.

    Параметры
    ----------
    run_dir : Path
        Путь к директории запуска.
    period : {'train', 'validation', 'test'}
        Период для оценки.
    epoch : int, необязательно
        Указать конкретную эпоху для использования. По умолчанию используются веса последней эпохи.
    gpu : int, необязательно
        ID GPU для использования. Переопределяет параметр 'device' в конфиге. Значение меньше нуля — CPU.
        Не используйте этот аргумент, если хотите использовать устройство, указанное в конфиге, например MPS.
    """
    config = Config(run_dir / "config.yml")


    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_evaluation(cfg=config, run_dir=run_dir, epoch=epoch, period=period)


if __name__ == "__main__":
    _main()
