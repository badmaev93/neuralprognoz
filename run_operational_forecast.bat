@ECHO OFF
CHCP 65001 > NUL

ECHO =====================================================
ECHO     Оперативное прогнозирование NeuralPrognoz
ECHO =====================================================
ECHO.

REM Activate conda environment
ECHO Активация окружения neuralprognoz...
CALL conda activate neuralprognoz

IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Не удалось активировать окружение neuralprognoz
    PAUSE
    EXIT /B 1
)

REM Set Python path
SET PYTHONPATH=%CD%

ECHO.
ECHO Запуск оперативного прогнозирования...
ECHO Режим: каждые 15 дней модель получает фактические данные
ECHO       и делает прогноз на следующие 15 дней
ECHO.

python scripts\operational_forecast.py

IF %ERRORLEVEL% EQU 0 (
    ECHO.
    ECHO =====================================================
    ECHO Оперативное прогнозирование завершено успешно!
    ECHO =====================================================
    ECHO.
    ECHO Результаты сохранены в папке runs\latest_run\
    ECHO - operational_forecast_results.csv
    ECHO - operational_forecast_metrics.csv
    ECHO.
) ELSE (
    ECHO.
    ECHO ERROR: Оперативное прогнозирование завершилось с ошибкой
    ECHO Код ошибки: %ERRORLEVEL%
)

ECHO.
PAUSE






