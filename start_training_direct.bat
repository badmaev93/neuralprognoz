@ECHO OFF
CHCP 866 > NUL

ECHO #####################################################
ECHO #                                                   #
ECHO #     NeuralPrognoz прямой скрипт обучение)         #
ECHO #        (без     conda run wrapper)                #
ECHO #                                                   #
ECHO #####################################################

REM --- Активировать conda напрямую  ---
ECHO.
ECHO активация окружения neuralprognoz ...

CALL C:\ProgramData\miniconda3\Scripts\activate.bat neuralprognoz

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ОШИБКА: не удалось активировать neuralprognoz 
    ECHO Попвтка исп. другого пути conda path...
    CALL C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat neuralprognoz
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ОШИБКА: Не удалось активировать environment. Проверьте установку conda.
        PAUSE
        EXIT /B 1
    )
)

REM --- Установка Python путей для локальных модулей ---
SET PYTHONPATH=%CD%

REM --- Проверка версии Python
ECHO.
ECHO Python version:
python --version

REM --- Проверить наличие необходимых файлов
IF NOT EXIST "np_run.py" (
    ECHO.
    ECHO ОШИБКА: np_run.py не найден!
    PAUSE
    EXIT /B 1
)

IF NOT EXIST "configs\train_model.yml" (
    ECHO.
    ECHO ОШИБКА: configs\train_model.yml не найден!
    PAUSE
    EXIT /B 1
)

REM --- Шаг 1. Обучение
ECHO.
ECHO #####################################################
ECHO #               ШАГ 1: ОБУЧЕНИЕ                     #
ECHO #####################################################
ECHO.
ECHO Начало обучения с конфигурацией: configs\train_model.yml
ECHO Это может занять много времени...
ECHO.

python np_run.py train --config-file configs\train_model.yml

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ОШИБКА: обучение не выполнено с кодом ошибки %ERRORLEVEL%
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO Обучение успешно завершено!

REM --- Шаг 2. Поиск каталог последнего запуска
ECHO.
ECHO #####################################################
ECHO #              ШАГ 2: ОЦЕНКА                        #
ECHO #####################################################
ECHO.
ECHO Поиск папки последнего запуска ...

SET LATEST_RUN_DIR=
FOR /f "delims=" %%i IN ('dir /b /ad /o-d runs\*') DO (
    IF EXIST "runs\%%i\config.yml" (
        SET LATEST_RUN_DIR=runs\%%i
        GOTO :found_run
    )
)

:found_run
IF NOT DEFINED LATEST_RUN_DIR (
    ECHO EОШИБКА: Не найден каталог запуска с config.yml
    PAUSE
    EXIT /B 1
)

ECHO Найдено: %LATEST_RUN_DIR%
ECHO.
ECHO Начало оценки...

python np_run.py evaluate --run-dir %LATEST_RUN_DIR%

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ОШИБКА: Оценка не удалась в кодом ошибки %ERRORLEVEL%
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO #####################################################
ECHO #                                                   #
ECHO #         ЦИКЛ ОБУЧЕНИЯ ЗАВЕРШЕН!                   #
ECHO #                                                   #
ECHO #     Результаты сохранены в %LATEST_RUN_DIR%       #
ECHO #                                                   #
ECHO #####################################################
ECHO.

PAUSE

