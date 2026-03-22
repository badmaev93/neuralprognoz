import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List

import matplotlib as mpl
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from neuralprognoz.__about__ import __version__
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.logging_utils import get_git_hash, save_git_diff


class Logger(object):
    """Класс, который регистрирует запуски в TensorBoard и сохраняет графики на диске.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """

    def __init__(self, cfg: Config):
        self._train = True
        self.log_interval = cfg.log_interval
        self.log_dir = cfg.run_dir
        self._img_log_dir = cfg.img_log_dir

        # получить хэш коммита git, если папка является репозиторием git
        cfg.update_config({'commit_hash': get_git_hash()})

        # сохранить git diff в файл, если ветка грязная
        if cfg.save_git_diff:
            save_git_diff(cfg.run_dir)

        # Кроме того, версия пакета хранится в конфигурации
        cfg.update_config({"package_version": __version__})

        # сохранить копию конфигурации в папке run
        cfg.dump_config(folder=self.log_dir)

        self.epoch = 0
        self.update = 0
        self._metrics = defaultdict(list)
        self.writer = None

    @property
    def tag(self):
        return "train" if self._train else "valid"

    def train(self) -> 'Logger':
        """Установите регистрацию для периода обучения.

        Возвращает
        -------
        Logger
            Экземпляр Logger, переведенный в режим обучения.
        """
        self._train = True
        return self

    def valid(self) -> 'Logger':
        """Установить ведение журнала на период проверки.

        Возвращает
        -------
        Logger
            Экземпляр Logger, установленный в режим валидации
        """
        self._train = False
        return self

    def start_tb(self):
        """ Начать ведение журнала Tensorboard. """
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def stop_tb(self):
        """ Остановить ведение журнала Tensorboard. """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_figures(self, figures: List[mpl.figure.Figure], freq: str, preamble: str = ""):
        """Записывать данные matplotlib на диск.

        Параметры
        ----------
        figures : List[mpl.figure.Figure]
            Список графиков для сохранения.
        freq : str
            Частота прогнозов графика
        preamble : str, optional
            Префикс названий файлов графиков.
        """
        if self.writer is not None:
            self.writer.add_figure(f'validation/timeseries/{freq}', figures, global_step=self.epoch)

        for idx, figure in enumerate(figures):
            figure.savefig(Path(self._img_log_dir, preamble + f'_freq{freq}_epoch{self.epoch}_{idx + 1}'), dpi=300)

    def log_step(self, **kwargs):
        """Лог результатов одного шага в течение эпохи.

        Параметры
        ----------
        **kwargs
            Пары ключ-значение названий и значений метрик.
        """
        for k, v in kwargs.items():
            self._metrics[k].append(v)

        if not self._train:
            return

        self.update += 1

        if self.log_interval <= 0 or self.writer is None:
            return

        if self.update % self.log_interval == 0:
            tag = self.tag
            for k, v in kwargs.items():
                self.writer.add_scalar('/'.join([tag, k]), v, self.update)

    def summarise(self) -> Union[float, Dict[str, float]]:
        """"Лог результатов всей эпохи обучения или проверки.

        Возвращает
        -------
        Union[float, Dict[str, float]]
            Средние потери, если обучение суммируется. Иначе - словарь, сопоставляющий названия метрик со средними значениями метрик.
        """
        value = {}
        # суммировать статистику эпохи обучения
        if self._train:
            self.epoch += 1

            # итоги обучения
            for k, v in self._metrics.items():
                mean = np.nanmean(v) if v else np.nan
                value[f'avg_{k}'] = mean

                if self.writer is not None:
                    self.writer.add_scalar('/'.join([self.tag, f'avg_{k}']), mean, self.epoch)

        # итоги валидации
        else:
            for k, v in self._metrics.items():
                if v and isinstance(v[0], tuple):
                    # Единственный передаваемый кортеж — это потеря прни валидации по каждому бассейну, которая представляет собой список кортежей, где
                    # каждый элемент определяется как (потери в бассейне, количество батчей). Суммарный показатель по бассейнам
                    # взвешенных по количеству батчей на бассейн, для приблизительного вычисления потерь при обучении.
                    v_not_nan = [(loss, samples) for loss, samples in v if not np.isnan(loss)]
                    num_samples = sum(samples for _, samples in v_not_nan)
                    if num_samples > 0:
                        weighted_loss = sum(loss * samples / num_samples for loss, samples in v_not_nan)
                    else:
                        weighted_loss = np.nan
                    value[f'avg_{k}'] = weighted_loss
                    if self.writer is not None:
                        self.writer.add_scalar('/'.join([self.tag, f'avg_{k}']), weighted_loss, self.epoch)
                else:
                    # Все остальные метрики представляют собой списки значений с плавающей точкой.
                    means = np.nanmean(v) if v else np.nan
                    medians = np.nanmedian(v) if v else np.nan
                    value[k] = medians
                    if self.writer is not None:
                        self.writer.add_scalar('/'.join([self.tag, f'mean_{k.lower()}']), means, self.epoch)
                        self.writer.add_scalar('/'.join([self.tag, f'median_{k.lower()}']), medians, self.epoch)

        # очистить буфер
        self._metrics = defaultdict(list)

        return value
