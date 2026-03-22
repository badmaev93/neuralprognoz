class NoTrainDataError(Exception):
    """Возникает, когда бассейн не содержит действительных выборок в тренировочный период."""

class NoEvaluationDataError(Exception):
    """Возникает, когда бассейн не содержит действительных выборок в периоды валидации или тестирования.""" 

class AllNaNError(Exception):
    """Исключение генерируется функциями calculate_metrics или calculate_all_metrics, если все наблюдения или симуляции содержат NaN-значения. """
