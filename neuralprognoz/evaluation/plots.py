from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def percentile_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Построение временного ряда наблюдаемых значений с 3 конкретными доверительными интервалами (т.е.: 25-75, 10-90, 5-95).

    Параметры
    ----------
    y : np.ndarray
        Массив наблюдаемых значений.
    y_hat : np.ndarray
        Массив симулированных значений, где последний размер содержит выборки для каждого временного шага.
    title : str, optional
        Заголовок графика.

    Возвращает
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        График доверительных интервалов.
    """
    fig, ax = plt.subplots()

    y_median = np.median(y_hat, axis=-1).flatten()
    y_25 = np.percentile(y_hat, 25, axis=-1).flatten()
    y_75 = np.percentile(y_hat, 75, axis=-1).flatten()
    y_10 = np.percentile(y_hat, 10, axis=-1).flatten()
    y_90 = np.percentile(y_hat, 90, axis=-1).flatten()
    y_05 = np.percentile(y_hat, 5, axis=-1).flatten()
    y_95 = np.percentile(y_hat, 95, axis=-1).flatten()

    x = np.arange(len(y_05))

    ax.fill_between(x, y_05, y_95, color='#35B779', label='05-95 PI')
    ax.fill_between(x, y_10, y_90, color='#31688E', label='10-90 PI')
    ax.fill_between(x, y_25, y_75, color="#440154", label='25-75 PI')
    ax.plot(y_median, '-', color='red', label="median")
    ax.plot(y.flatten(), '--', color='black', label="observed")
    ax.legend()
    ax.set_title(title)

    return fig, ax


def regression_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Построение временного ряда наблюдаемых и симулированных значений.

    Параметры
    ----------
    y : np.ndarray
        Массив наблюдаемых значений.
    y_hat : np.ndarray
        Массив симулированных значений.
    title : str, optional
        Заголовок графика.

    Возвращает
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        График временного ряда наблюдаемых и симулированных значений.
    """

    fig, ax = plt.subplots()

    ax.plot(y.flatten(), label="observed", lw=1)
    ax.plot(y_hat.flatten(), label="simulated", alpha=.8, lw=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title(title)

    return fig, ax


def uncertainty_plot(y: np.ndarray, y_hat: np.ndarray, title: str = '') -> Tuple[mpl.figure.Figure, np.ndarray]:
    """Построение графика вероятности вместе с гидрографом с доверительными интервалами симуляции.
    
    Сам график вероятности аналогичен калибровочному графику для задач классификации. График сравнивает 
    теоретические квантили оцененных условных распределений (во времени) с соответствующими относительными 
    эмпирическими частотами. 
    График вероятности часто также называют диаграммой преобразования интеграла вероятности, Q-Q диаграммой или предиктивной 
    Q-Q диаграммой. 
    

    Параметры
    ----------
    y : np.ndarray
        Массив наблюдаемых значений.
    y_hat : np.ndarray
        Массив симулированных значений.
    title : str, optional
        Заголовок графика.

    Возвращает
    -------
    Tuple[mpl.figure.Figure, np.ndarray]
        График неопределенности.
    """

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3), gridspec_kw={'width_ratios': [4, 5]})

    # берем только часть y, чтобы лучше рассмотреть
    y_long = y[:, -1].flatten()
    y_hat_long = y_hat[:, -1, :].reshape(y_long.shape[0], -1)
    x_bnd = np.arange(0, 400)
    y_bnd_len = len(x_bnd)

    # гидрограф:
    y_r = [0, 0, 0, 0, 0, 0]  # используется позже для графика вероятности
    quantiles = [0.9, 0.80, 0.50, 0.20, 0.1]
    labels_and_colors = {
        'labels': ['05-95 PI', '10-90 PI', '25-75 PI', '40-60 PI', '45-55 PI'],
        'colors': ['#FDE725', '#8FD744', '#21908C', '#31688E', '#443A83']
    }
    for idx in range(len(quantiles)):
        lb = round(50 - (quantiles[idx] * 100) / 2)
        ub = round(50 + (quantiles[idx] * 100) / 2)
        y_lb = np.percentile(y_hat_long[x_bnd, :], lb, axis=-1).flatten()
        y_ub = np.percentile(y_hat_long[x_bnd, :], ub, axis=-1).flatten()
        y_r[idx] = np.sum(((y_long[x_bnd] > y_lb) * (y_long[x_bnd] < y_ub))) / y_bnd_len
        if idx <= 3:
            axs[1].fill_between(x_bnd,
                                y_lb,
                                y_ub,
                                color=labels_and_colors['colors'][idx],
                                label=labels_and_colors['labels'][idx])

    y_median = np.median(y_hat_long, axis=-1).flatten()
    axs[1].plot(x_bnd, y_median[x_bnd], '-', color='red', label="median")
    axs[1].plot(x_bnd, y_long[x_bnd], '--', color='black', label="observed")
    axs[1].legend(prop={'size': 5})
    axs[1].set_ylabel("value")
    axs[1].set_xlabel("time index")
    # график вероятности:
    quantiles = np.arange(0, 101, 5)
    y_r = quantiles * 0.0
    for idx in range(len(y_r)):
        ub = quantiles[idx]
        y_ub = np.percentile(y_hat_long[x_bnd, :], ub, axis=-1).flatten()
        y_r[idx] = np.sum(y_long[x_bnd] < y_ub) / y_bnd_len

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].plot(quantiles / 100, y_r, 'ro', ms=3.0)
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.grid(color='#ECECEC', linestyle='dashed')
    axs[0].xaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].yaxis.set_ticks(np.arange(0, 1, 0.2))
    axs[0].set_xlabel("теоретическая частота квантиля")
    axs[0].set_ylabel("количество")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    return fig, axs
