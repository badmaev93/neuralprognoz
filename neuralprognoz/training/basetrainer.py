import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralprognoz.training.loss as loss
from neuralprognoz.datasetzoo import get_dataset
from neuralprognoz.datasetzoo.basedataset import BaseDataset
from neuralprognoz.datautils.utils import load_basin_file, load_scaler
from neuralprognoz.evaluation import get_tester
from neuralprognoz.evaluation.tester import BaseTester
from neuralprognoz.modelzoo import get_model
from neuralprognoz.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralprognoz.training.logger import Logger
from neuralprognoz.utils.config import Config
from neuralprognoz.utils.logging_utils import setup_logging
from neuralprognoz.training.earlystopper import EarlyStopper
from neuralprognoz.training.visualization_hook import create_validation_plots

LOGGER = logging.getLogger(__name__)


class BaseTrainer(object):
    """Класс для обучечения модели.

    Параметры
    ----------
    cfg : Config
        Конфигурвция запуска
    """

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._early_stopping = cfg.early_stopping
        self._patience_early_stopping = cfg.patience_early_stopping
        self._minimum_epochs_before_early_stopping = cfg.minimum_epochs_before_early_stopping
        self._dynamic_learning_rate = cfg.dynamic_learning_rate
        self._patience_dynamic_learning_rate = cfg.patience_dynamic_learning_rate
        self._factor_dynamic_learning_rate = cfg.factor_dynamic_learning_rate

        # загрузить список бассейнов и добавить количесто в конфиг
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # проверить с какой эпохи начинается обучение
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Созданы директории: {self.cfg.run_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Продолжение обучения следующего запуска: {self.cfg.base_run_dir}")

        if self.cfg.is_finetuning:
            LOGGER.info(f"### Запуск файнтюнинга с помощью следующей предобученной модели: {self.cfg.base_run_dir}")

        LOGGER.info(f"### Запуск конфигураций для {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        return DataLoader(ds,
                          batch_size=self.cfg.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.num_workers,
                          pin_memory=self.device.type != "cpu",
                          collate_fn=ds.collate_fn)

    def _freeze_model_parts(self):
        # заморозка всех весов модели
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # зарморозка параметров, указанных в конфигурации, как настраиваемых параметров
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # если это был не список, то это должен быть словарь
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Не удалось разрешить для файнтюнинга следующие части модуля: {unresolved_modules}")

    def initialize_training(self):
        """Инициализация класса обучения.

         Метод загрузит модель, инициализирует потери, регуляризацию, оптимизатор, набор данных и загрузчик данных,
         лог tensorboard и класс Tester.

         При вызове в контексте `continue_training` эта модель также восстановит состояние модели и оптимизатора.
        """
        if self.cfg.is_finetuning:
            # Загрузить scaler из предобученной модели.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Инициализировать датасет до загрузки модели.
        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Датасет не содержит сэмплов.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)
        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Запуск обучения с точки сохранения. {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device, weights_only=True))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            # по умолчанию для точной настройки используется последнее состояние модели
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Запуск обучения с чекпоинта {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device, weights_only=True))

        # Заморозмить детали модели из предобученной модели.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # добавить возможные условия регуляризации к функции потерь.
        self._set_regularization()

        # восстановление состояния оптимизатора и модели при продолжении обучения
        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # установить эпоху и счетчик итераций для продолжения с выбранной точки.
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Валидация установлена на валидацию каждые {self.cfg.validate_every} эпох(и), но ",
                    "'validate_n_random_basins' не установлено или равно 0. Валидация будет проходить по всему набору валидаций."
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values).to(self.device)

    def train_and_validate(self):
        """Обучение и валидация модели.
        Обучение модели в течение количества эпох, указанного в конфигурации запуска, и выполнение валидации после каждой
        ``validate_every`` эпохи. Состояние модели и оптимизатора сохраняется после каждой ``save_weights_every`` эпохи.
        """
        # Инициализация early stopping
        best_epoch = None
        best_loss = float('inf')
        
        if self._early_stopping:
            if self.cfg.is_continue_training:
                LOGGER.warning("Состояние early stopping сброшено.")   
            early_stopper = EarlyStopper(patience=self._patience_early_stopping, min_delta=0.0001)
        
        # Инициализация динамического learning rate
        if self._dynamic_learning_rate:
            if self.cfg.is_continue_training:
                LOGGER.warning("Состояние scheduler сброшено")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self._factor_dynamic_learning_rate, 
                patience=self._patience_dynamic_learning_rate
            )
        
        # Основной цикл обучения
        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            # Обновление learning rate (если не динамический)
            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    LOGGER.info(f"Скорость обучения установлена на {self.cfg.learning_rate[epoch]}")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.cfg.learning_rate[epoch]
            
            # Тренировка одной эпохи
            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Эпоха {epoch}, средние потери: {loss_str}")
            
            # Сохранение весов
            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)
            
            # Валидация
            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                # Запуск валидации
                validation_results = self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    save_all_output=self.cfg.save_all_output,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid()
                )
                
                # Получение метрик валидации
                valid_metrics = self.experiment_logger.summarise()
                current_loss = valid_metrics['avg_total_loss']
                
                # Логирование метрик
                print_msg = f"Эпоха {epoch} средние потери при валидации: {current_loss:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Медианные метрики валидации: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
                LOGGER.info(print_msg)
                
                # Отслеживание лучшей модели
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    LOGGER.info(f"✓ Новая лучшая модель на эпохе {epoch} (loss: {best_loss:.5f})")
                
                # === ГЕНЕРАЦИЯ ГРАФИКОВ ===
                if self.cfg.save_validation_results and hasattr(self.cfg, 'img_log_dir'):
                    try:
                        create_validation_plots(
                            results_dict=validation_results,
                            epoch=epoch,
                            img_log_dir=self.cfg.img_log_dir
                        )
                    except Exception as e:
                        LOGGER.warning(f"Не удалось создать визуализацию для эпохи {epoch}: {e}")
                
                # Проверка early stopping
                if self._early_stopping and epoch > self._minimum_epochs_before_early_stopping:
                    if early_stopper.check_early_stopping(current_loss):
                        LOGGER.info(f"Early stopping сработал на эпохе {epoch} (loss: {current_loss:.5f})")
                        LOGGER.info(f"Лучшая модель была на эпохе {best_epoch} (loss: {best_loss:.5f})")
                        
                        # ВАЖНО: Загружаем веса лучшей модели
                        if best_epoch is not None and best_epoch != epoch:
                            best_weights_path = self.cfg.run_dir / f"model_epoch{best_epoch:03d}.pt"
                            if best_weights_path.exists():
                                LOGGER.info(f"Загрузка весов лучшей модели с эпохи {best_epoch}")
                                weight_file = torch.load(best_weights_path, map_location=self.device, weights_only=True)

                                # Проверяем формат файла (словарь или чистые веса)
                                if isinstance(weight_file, dict) and 'model_state_dict' in weight_file:
                                    self.model.load_state_dict(weight_file['model_state_dict'])
                                else:
                                    # Если сохранен просто state_dict (как делает _save_weights_and_optimizer)
                                    self.model.load_state_dict(weight_file)

                                # Сохраняем финальную модель
                                final_path = self.cfg.run_dir / "model_final.pt"
                                # Тут сохраняем правильно, с метаданными
                                torch.save({
                                    'model_state_dict': self.model.state_dict(),
                                    'epoch': best_epoch,
                                    'loss': best_loss
                                }, final_path)
                                LOGGER.info(f"Финальная модель сохранена: {final_path}")
                            else:
                                LOGGER.warning(f"Веса эпохи {best_epoch} не найдены, используется текущая модель")
                        
                        break
                
                # Обновление learning rate (если динамический)
                if self._dynamic_learning_rate:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    scheduler.step(current_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        LOGGER.info(f"Learning rate изменен: {old_lr:.6f} → {new_lr:.6f}")
        
        # Финальное логирование
        if best_epoch is not None:
            LOGGER.info("="*60)
            LOGGER.info(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
            LOGGER.info(f"Лучшая модель: эпоха {best_epoch}, loss: {best_loss:.5f}")
            LOGGER.info("="*60)
        
        # Закрытие tensorboard
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob('model_epoch*.pt')))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Продолжение обучения с эпохи {int(epoch)}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device, weights_only=True))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # ручка панели процессов
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Эпоха {epoch}')

        # Выполнять итерации батчами по обучающему набору
        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device, non_blocking=True) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device, non_blocking=True)

            # применить возможную предобработку к батчу перед прямым проходом
            data = self.model.pre_model_hook(data, is_train=True)

            # получить прогнозы
            predictions = self.model(data)

            if self.noise_sampler_y is not None:
                for key in filter(lambda k: 'y' in k, data.keys()):
                    noise = self.noise_sampler_y.sample(data[key].shape)
                    # убедиться что добавлен шум близкий к нулю к изначально близким к нулю целям
                    data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device, non_blocking=True)

            loss, all_losses = self.loss_obj(predictions, data)

            # early stop обучения если потери NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Потери NaN {nan_count} раз в строке. Остановка обучения.")
                LOGGER.warning(f"Потери Nan; игнорирование шага. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0

                # удаление старых градиентов
                self.optimizer.zero_grad(set_to_none=True)

                # получить градиенты
                loss.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                # обновить веса
                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})
    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        # исправить случайные сиды для различных пакетов
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"Не обнаружено GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS устройство недоступно.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        LOGGER.info(f"### Устройство {self.device} будет использовано для обучения")

    def _create_folder_structure(self):
        # создать как подкаталог в каталоге выполнения базового запуска
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"

            # сохранить каталог базового запуска для более легкого доступа при загрузке весов
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # создание новой структуры папок
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

            # Если каталог для запусков не указан, папка runs будет создана в текущем рабочем каталоге.
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # создать каталог + нужный подкаталог
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"Папка уже есть {self.cfg.run_dir}")
        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
