from typing import Sequence

import numpy as np


class RatingCurve(object):
    """Класс для оценки кривых расхода воды на основе данных уровня воды и расхода, а также преобразования уровня воды в расход и наоборот.
            
    Параметры
    ----------
    stages : Sequence
        Данные уровней воды для построения кривой расходов
    discharges : Sequence
        Данные расходов воды для построения кривой расходов 
    move_stages_to_zero : bool, optional
        Опция учета любого смещения в данных уровня воды. Автоматически
        устанавливает минимальный измеренный уровень воды в ноль 
    """

    def __init__(self, stages: Sequence[float], discharges: Sequence[float], move_stages_to_zero: bool = True):

        # проверка входных данных
        if len(stages) != len(discharges):
            raise ValueError("Последовательности 'stages' и 'discharges' должны иметь одинаковую длину.")

        self.stages = np.array(stages, dtype=np.float32)
        self.discharges = np.array(discharges, dtype=np.float32)

        # В интерполяции ось X представляет собой уровень воды (stage), а ось Y — расход (discharge).
        if move_stages_to_zero:
            use_stages = self.stages - np.min(self.stages)
            self.zero_stages = np.min(self.stages)
        else:
            use_stages = self.stages
            self.zero_stages = False

        self.coeffs = np.polyfit(use_stages, self.discharges, deg=2)

    def stage_to_discharge(self, stage: np.ndarray) -> np.ndarray:
        """Преобразование уровня воды в расход.
            
        Параметры
        ----------
        stage : np.ndarray
            Массив данных уровня воды для преобразования в расход.
    
        Возвращает
        -------
        np.ndarray
            Оценённый расход.
        """

        if self.zero_stages:
            stage -= self.zero_stages

        return self.coeffs[2] + stage * self.coeffs[1] + stage**2 * self.coeffs[0]

    def discharge_to_stage(self, discharge: np.ndarray) -> np.ndarray:
        """Преобразовать расход в уровень воды.
            
        Параметры
        ----------
        discharge : np.ndarray
            Массив данных расхода для преобразования в уровень воды. 
  
        Возвращает
        -------
        np.ndarray
            Оценённый уровень воды.
        """

        # инициализация хранилища
        stage = np.full(discharge.shape, np.nan)
        solutions = np.full([discharge.shape[0], 2], np.nan)

        centered_bias = self.coeffs[2] - discharge
        radicand = self.coeffs[1]**2 - 4 * self.coeffs[0] * centered_bias
        radicand[np.isnan(radicand)] = -9999
        mask = radicand >= 0

        solutions[mask, 0] = (-self.coeffs[1] + np.sqrt(radicand[mask])) / (2 * self.coeffs[0])
        solutions[mask, 1] = (-self.coeffs[1] - np.sqrt(radicand[mask])) / (2 * self.coeffs[0])

        best_idx = np.argmin(np.abs(solutions[mask] - np.expand_dims(discharge[mask], -1)), axis=1)

        stage[mask] = solutions[mask, best_idx]

        if self.zero_stages:
            stage += self.zero_stages

        return stage
