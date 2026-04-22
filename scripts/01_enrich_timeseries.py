import logging
from pathlib import Path
import pandas as pd
import xarray as xr
from tqdm import tqdm
import numpy as np

try:
    from neuralprognoz.datautils import climateindices, pet, penman_monteith, wind_conversion
except ImportError:
    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from neuralprognoz.datautils import climateindices, pet, penman_monteith, wind_conversion


# --- НАСТРОЙКИ СКРИПТА ---

# 1. Пути к данным (относительно корня проекта)
PROJECT_ROOT = Path(__file__).parent.parent
TIME_SERIES_DIR = PROJECT_ROOT / "data" / "generic_dataset" / "time_series"
ATTRIBUTES_FILE = PROJECT_ROOT / "data" / "generic_dataset" / "attributes" / "attributes.csv"

# 2. Сопоставление имен переменных как они называются в .nc и .csv файлах
VAR_MAP = {
    "temperature_min": "temperature_min",
    "temperature_max": "temperature_max", 
    "temperature_mean": "temperature_mean",
    "solar_radiation": "solar_radiation",
    "precipitation": "precipitation",
    "dewpoint": "dewpoint", 
    "wind_speed_10m": "wind_speed_10m",
    "wind_speed_2m": "wind_speed_2m",
    "wind_u": "wind_u",
    "wind_v": "wind_v", 
    "albedo": "albedo",
    "pressure": "pressure"
}

ATTR_MAP = {
    # Для расчета PET
    "lat": "mean_lat_deg",
    "elev": "mean_elevation_m",
    # Для конвертации стока (если нужно)
    "area": "area_km2",
}

METADATA_MAP = {
    'pet_penman_monteith': {
        'long_name': 'Потенциальная эвапотранспирация (Пенман-Монтейт)', 
        'units': 'mm/day',
        'description': 'Рассчитано по методу FAO56 Пенман-Монтейт'
    },
    'pet_priestley_taylor': {
        'long_name': 'Потенциальная эвапотранспирация (Пристли-Тейлор)', 
        'units': 'mm/day',
        'description': 'Рассчитано по методу Пристли-Тейлора с альфа=1.22 (калиброван для субарктического климата)'
    },
    'p_mean_dyn': {
        'long_name': 'Динамическое среднее количество осадков',
        'units': 'mm/day',
        'description': 'Скользящее среднее количество осадков за 30 дней'
    },
    'pet_mean_dyn': {
        'long_name': 'Динамическое среднее значение потенциальной эвапотранспирации',
        'units': 'mm/day',
        'description': 'Скользящее среднее PET за 30 дней'
    },
    'aridity_dyn': {
        'long_name': 'Динамический индекс засушливости (PET/P)',
        'units': '-',
        'description': 'Отношение потенциальной эвапотранспирации к осадкам'
    },
    't_mean_dyn': {
        'long_name': 'Динамическое среднее значение температуры',
        'units': '°C',
        'description': 'Скользящее среднее температуры за 30 дней'
    },
    'frac_snow_dyn': {
        'long_name': 'Динамическая доля осадков в виде снега',
        'units': '-',
        'description': 'Доля осадков, выпадающих в виде снега (температура < 0°C)'
    },
    'high_prec_freq_dyn': {
        'long_name': 'Динамическая частота сильных осадков',
        'units': '-',
        'description': 'Частота дней с осадками > 10 мм/день за 30 дней'
    },
    'high_prec_dur_dyn': {
        'long_name': 'Динамическая продолжительность сильных осадков',
        'units': 'дни',
        'description': 'Продолжительность периодов с осадками > 10 мм/день'
    },
    'low_prec_freq_dyn': {
        'long_name': 'Динамическая частота слабых осадков',
        'units': '-',
        'description': 'Частота дней с осадками < 1 мм/день за 30 дней'
    },
    'low_prec_dur_dyn': {
        'long_name': 'Динамическая продолжительность слабых осадков',
        'units': 'дни',
        'description': 'Продолжительность периодов с осадками < 1 мм/день'
    },
    'precipitation_lag_1d': {
        'long_name': 'Осадки с лагом 1 день',
        'units': 'mm/day',
        'description': 'Значение осадков предыдущего дня'
    },
    'precipitation_lag_2d': {
        'long_name': 'Осадки с лагом 2 дня',
        'units': 'mm/day',
        'description': 'Значение осадков за 2 дня назад'
    },
    'precipitation_rolling_sum_7d': {
        'long_name': 'Сумма осадков за 7 дней',
        'units': 'mm',
        'description': 'Скользящая сумма осадков за 7 дней'
    },
    'precipitation_rolling_sum_30d': {
        'long_name': 'Сумма осадков за 30 дней',
        'units': 'mm',
        'description': 'Скользящая сумма осадков за 30 дней'
    }
}

# 3. Настройки для расчетов
CALCULATE_PET = True
PET_METHOD = "both"  # Доступные методы: "penman-monteith", "priestley-taylor", "both"
PET_PRIESTLEY_TAYLOR_ALPHA = 1.22  # Коэффициент альфа для Пристли-Тейлора (калиброван для субарктического климата Колымы)
CALCULATE_CLIMATE_INDICES = True
CLIMATE_INDICES_WINDOW = 730 # Длина окна в днях для расчета скользящих климатических индексов


def setup_logging():
    """Настраивает логирование для вывода информации в консоль."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def calculate_snow_fraction(temperature, precipitation):
    """
    Рассчитывает долю осадков в виде снега на основе температуры.
    
    Args:
        temperature: температура воздуха (°C)
        precipitation: количество осадков (mm/day)
    
    Returns:
        Доля осадков в виде снега (0-1)
    """
    # Если температура выше 0°C, снега нет
    # Если температура ниже -5°C, все осадки в виде снега
    # Между 0°C и -5°C - линейная интерполяция
    snow_fraction = np.where(temperature >= 0, 0,
                    np.where(temperature <= -5, 1,
                    1 + temperature / 5))
    
    # Применяем только к дням с осадками
    snow_fraction = np.where(precipitation > 0, snow_fraction, 0)
    
    return snow_fraction

def calculate_precipitation_statistics(precipitation, window=30):
    """
    Рассчитывает статистики осадков для динамических индексов.
    
    Args:
        precipitation: временной ряд осадков
        window: размер окна для расчета (дни)
    
    Returns:
        Словарь с рассчитанными индексами
    """
    if hasattr(precipitation, 'to_series'):
        precip_series = precipitation.to_series()
    else:
        precip_series = pd.Series(precipitation)
    
    # Высокие осадки (> 10 мм/день)
    high_prec = precip_series > 10
    high_prec_freq = high_prec.rolling(window=window, center=True).mean()
    
    # Слабые осадки (< 1 мм/день)
    low_prec = precip_series < 1
    low_prec_freq = low_prec.rolling(window=window, center=True).mean()
    
    # Продолжительность периодов с высокими осадками
    high_prec_dur = high_prec.astype(int).rolling(window=window, center=True).sum()
    
    # Продолжительность периодов со слабыми осадками
    low_prec_dur = low_prec.astype(int).rolling(window=window, center=True).sum()
    
    return {
        'high_prec_freq_dyn': high_prec_freq.values,
        'low_prec_freq_dyn': low_prec_freq.values,
        'high_prec_dur_dyn': high_prec_dur.values,
        'low_prec_dur_dyn': low_prec_dur.values
    }

def enrich_timeseries():
    """
    Основная функция для обогащения временных рядов.
    Проходит по всем .nc файлам в папке time_series, рассчитывает PET и/или
    климатические индексы и перезаписывает файлы с новыми переменными.
    """
    logging.info("Запуск скрипта для обогащения временных рядов...")

    # --- Загрузка атрибутов ---
    try:
        attributes_df = pd.read_csv(ATTRIBUTES_FILE, index_col="basin_id", dtype={'basin_id': str})
        logging.info(f"Атрибуты успешно загружены из {ATTRIBUTES_FILE}")
    except FileNotFoundError:
        logging.error(f"Файл атрибутов не найден: {ATTRIBUTES_FILE}. Прерывание.")
        return
    except Exception as e:
        logging.error(f"Не удалось прочитать файл атрибутов: {e}. Прерывание.")
        return

    basin_files = sorted(list(TIME_SERIES_DIR.glob("*.nc")))
    if not basin_files:
        logging.warning(f"В папке {TIME_SERIES_DIR} не найдено .nc файлов.")
        return

    logging.info(f"Найдено {len(basin_files)} файлов .nc для обработки.")

    # --- Основной цикл обработки ---
    for basin_file in tqdm(basin_files, desc="Processing basins"):
        basin_id = basin_file.stem.replace('_raw', '')

        # 1. Загрузка атрибутов для текущего бассейна
        try:
            basin_attrs = attributes_df.loc[basin_id]
        except KeyError:
            logging.warning(f"Атрибуты для бассейна {basin_id} не найдены в {ATTRIBUTES_FILE}. Пропускаем.")
            continue

        # 2. Открытие NetCDF файла
        try:
            with xr.open_dataset(basin_file) as ds_temp:
                ds = ds_temp.load()
        except Exception as e:
            logging.error(f"Не удалось открыть или прочитать файл {basin_file}: {e}. Пропускаем.")
            continue

        file_modified = False

        # 3. Переименование time в date и добавление doy_sin/doy_cos
        if 'time' in ds.coords and 'date' not in ds.coords:
            try:
                logging.info(f"[{basin_id}] Переименование time в date и добавление doy_sin/doy_cos...")
                
                if ds['time'].dtype == 'datetime64[ns]':
                    dates = pd.to_datetime(ds['time'].values)
                    doy = dates.dayofyear.values
                else:
                    base_date = pd.Timestamp('1993-01-01')
                    dates = base_date + pd.to_timedelta(ds['time'].values, unit='D')
                    doy = dates.dayofyear.values
                
                doy_sin = np.sin(2 * np.pi * doy / 365.25)
                doy_cos = np.cos(2 * np.pi * doy / 365.25)
                
                ds = ds.rename({'time': 'date'})
                
                ds['doy_sin'] = (('date'), doy_sin)
                ds['doy_sin'].attrs = {
                    'long_name': 'Day of Year (sine)',
                    'units': '-',
                    'description': 'Sine of day of year for seasonal cycles'
                }
                
                ds['doy_cos'] = (('date'), doy_cos)
                ds['doy_cos'].attrs = {
                    'long_name': 'Day of Year (cosine)',
                    'units': '-',
                    'description': 'Cosine of day of year for seasonal cycles'
                }
                
                for var_name in ds.data_vars:
                    if 'time' in ds[var_name].dims:
                        ds[var_name] = ds[var_name].rename_dims({'time': 'date'})
                
                file_modified = True
                logging.info(f"[{basin_id}] Координата time переименована в date, добавлены doy_sin и doy_cos.")
                
            except Exception as e:
                logging.error(f"[{basin_id}] Ошибка при переименовании time в date: {e}. Пропускаем.")

        # 4. Расчет скорости ветра из компонент (если нет готовой переменной)
        if 'wind_speed_10m' not in ds.variables and 'wind_u' in ds.variables and 'wind_v' in ds.variables:
            try:
                logging.info(f"[{basin_id}] Расчет скорости ветра из компонент u и v...")
                
                wind_u = ds['wind_u'].values
                wind_v = ds['wind_v'].values
                
                wind_speed_10m = penman_monteith.calculate_wind_speed(wind_u, wind_v)
                
                ds['wind_speed_10m'] = (('date'), wind_speed_10m)
                ds['wind_speed_10m'].attrs = {
                    'long_name': 'Wind Speed at 10m',
                    'units': 'm/s',
                    'description': 'Calculated from u and v wind components'
                }
                
                file_modified = True
                logging.info(f"[{basin_id}] Скорость ветра на 10м успешно рассчитана из компонент.")
                
            except Exception as e:
                logging.error(f"[{basin_id}] Ошибка при расчете скорости ветра: {e}. Пропускаем.")

        # 4a. Расчет скорости ветра на 2м для Пенмана-Монтейта
        if 'wind_speed_2m' not in ds.variables and 'wind_speed_10m' in ds.variables:
            try:
                logging.info(f"[{basin_id}] Преобразование скорости ветра с 10м на 2м...")
                
                wind_speed_10m = ds['wind_speed_10m'].values
                
                dates = pd.to_datetime(ds['date'].values)
                doy = dates.dayofyear.values
                
                wind_speed_2m = wind_conversion.convert_wind_speed_10m_to_2m_seasonal(
                    wind_speed_10m, 
                    doy=doy,
                    land_use_type="agricultural"  #  тип землепользования
                )
                
                ds['wind_speed_2m'] = (('date'), wind_speed_2m)
                ds['wind_speed_2m'].attrs = {
                    'long_name': 'Wind Speed at 2m',
                    'units': 'm/s',
                    'description': 'Converted from 10m using logarithmic wind profile with seasonal roughness length (agricultural land use)'
                }
                
                roughness_seasonal = wind_conversion.calculate_seasonal_roughness_length(doy, "agricultural")
                ds['roughness_length'] = (('date'), roughness_seasonal)
                ds['roughness_length'].attrs = {
                    'long_name': 'Surface Roughness Length',
                    'units': 'm',
                    'description': 'Seasonal surface roughness length for agricultural land use'
                }
                
                file_modified = True
                logging.info(f"[{basin_id}] Скорость ветра на 2м успешно рассчитана.")
                
            except Exception as e:
                logging.error(f"[{basin_id}] Ошибка при преобразовании скорости ветра: {e}. Пропускаем.")

        # 5. Расчет PET по выбранному методу
        if CALCULATE_PET:
            try:
                base_vars = ['temperature_min', 'temperature_max', 'solar_radiation']
                missing_base = [var for var in base_vars if var not in ds.variables]
                
                if missing_base:
                    logging.warning(f"[{basin_id}] Отсутствуют базовые переменные для расчета PET: {missing_base}. Пропускаем.")
                    raise ValueError(f"Missing base variables: {missing_base}")
                
                t_min = ds['temperature_min'].values
                t_max = ds['temperature_max'].values
                s_rad = ds['solar_radiation'].values
                
                lat_deg = basin_attrs[ATTR_MAP["lat"]]
                elev_m = basin_attrs[ATTR_MAP["elev"]]
                
                dates = pd.to_datetime(ds['date'].values)
                doy = dates.dayofyear.values
                
                logging.info(f"[{basin_id}] Начинаем расчет PET по методу: {PET_METHOD}")
                
            except Exception as e:
                logging.error(f"[{basin_id}] Ошибка при подготовке данных для PET: {e}. Пропускаем.")
                continue
            
            # 5a. Метод Пристли-Тейлора
            if PET_METHOD in ["priestley-taylor", "both"] and 'pet_priestley_taylor' not in ds.variables:
                try:
                    logging.info(f"[{basin_id}] Расчет PET по методу Пристли-Тейлора...")
                    
                    # Проверяем наличие альбедо 
                    if 'albedo' in ds.variables:
                        logging.info(f"[{basin_id}] Используем улучшенную версию Пристли-Тейлора с альбедо...")
                        albedo = ds['albedo'].values
                        
                        pet_pt_values = pet.get_priestley_taylor_pet_with_albedo(
                            t_min=t_min,
                            t_max=t_max,
                            s_rad=s_rad,
                            lat=lat_deg,
                            elev=elev_m,
                            doy=doy,
                            albedo=albedo,
                            alpha=PET_PRIESTLEY_TAYLOR_ALPHA
                        )
                    else:
                        logging.info(f"[{basin_id}] Используем стандартную версию Пристли-Тейлора...")
                        pet_pt_values = pet.get_priestley_taylor_pet(
                            t_min=t_min,
                            t_max=t_max,
                            s_rad=s_rad,
                            lat=lat_deg,
                            elev=elev_m,
                            doy=doy
                        )
                    
                    ds['pet_priestley_taylor'] = (('date'), pet_pt_values)
                    ds['pet_priestley_taylor'].attrs = METADATA_MAP['pet_priestley_taylor']
                    
                    file_modified = True
                    logging.info(f"[{basin_id}] PET по методу Пристли-Тейлора успешно рассчитан.")
                    
                except Exception as e:
                    logging.error(f"[{basin_id}] Ошибка при расчете PET Пристли-Тейлора: {e}")
            
            # 5b. Метод Пенмана-Монтейта
            if PET_METHOD in ["penman-monteith", "both"] and 'pet_penman_monteith' not in ds.variables:
                try:
                    logging.info(f"[{basin_id}] Расчет PET по методу Пенмана-Монтейта...")
                    
                    # Для Пенмана-Монтейта нужны дополнительные переменные
                    pm_vars = ['dewpoint']
                    wind_vars = ['wind_speed_2m', 'wind_speed_10m']
                    
                    missing_pm = [var for var in pm_vars if var not in ds.variables]
                    has_wind = any(var in ds.variables for var in wind_vars)
                    
                    if missing_pm or not has_wind:
                        missing_info = []
                        if missing_pm:
                            missing_info.extend(missing_pm)
                        if not has_wind:
                            missing_info.append("wind_speed (2m or 10m)")
                        logging.warning(f"[{basin_id}] Отсутствуют переменные для Пенмана-Монтейта: {missing_info}. Пропускаем.")
                        raise ValueError(f"Missing PM variables: {missing_info}")
                    
                    t_dew = ds['dewpoint'].values
                    
                    if 'wind_speed_2m' in ds.variables:
                        wind_speed = ds['wind_speed_2m'].values
                        wind_height = "2m"
                        logging.info(f"[{basin_id}] Используем скорость ветра на 2м (оптимально для PET)")
                    else:
                        wind_speed = ds['wind_speed_10m'].values  
                        wind_height = "10m"
                        logging.info(f"[{basin_id}] Используем скорость ветра на 10м (будет преобразована в 2м)")
                    
                    use_enhanced_function = 'albedo' in ds.variables
                    
                    if use_enhanced_function:
                        logging.info(f"[{basin_id}] Используем улучшенную функцию PET с альбедо...")
                        albedo = ds['albedo'].values
                        pressure = ds['pressure'].values if 'pressure' in ds.variables else None
                        
                        if wind_height == "10m":
                            logging.info(f"[{basin_id}] Преобразуем скорость ветра с 10м на 2м...")
                            wind_speed_for_pet = wind_conversion.convert_wind_speed_10m_to_2m_seasonal(
                                wind_speed, doy=doy, land_use_type="agricultural"
                            )
                        else:
                            wind_speed_for_pet = wind_speed
                        
                        pet_pm_values = penman_monteith.fao56_penman_monteith_eto_with_albedo(
                            t_min=t_min,
                            t_max=t_max,
                            wind_speed_10m=wind_speed_for_pet,  # На самом деле это 2м
                            s_rad_wm2=s_rad,
                            lat_deg=lat_deg,
                            elev_m=elev_m,
                            doy=doy,
                            albedo=albedo,
                            tdew=t_dew,
                            pressure=pressure
                        )
                    else:
                        logging.info(f"[{basin_id}] Используем стандартную функцию PET...")
                        
                        if wind_height == "10m":
                            logging.info(f"[{basin_id}] Преобразуем скорость ветра с 10м на 2м...")
                            wind_speed_for_pet = wind_conversion.convert_wind_speed_10m_to_2m_seasonal(
                                wind_speed, doy=doy, land_use_type="agricultural"
                            )
                        else:
                            wind_speed_for_pet = wind_speed
                        
                        pet_pm_values = penman_monteith.fao56_penman_monteith_eto(
                            t_min=t_min,
                            t_max=t_max,
                            wind_speed_10m=wind_speed_for_pet,  # На самом деле это 2м
                            s_rad_wm2=s_rad,
                            lat_deg=lat_deg,
                            elev_m=elev_m,
                            doy=doy,
                            tdew=t_dew
                        )
                    
                    ds['pet_penman_monteith'] = (('date'), pet_pm_values)
                    ds['pet_penman_monteith'].attrs = METADATA_MAP['pet_penman_monteith']
                    
                    file_modified = True
                    logging.info(f"[{basin_id}] PET по методу Пенмана-Монтейта успешно рассчитан.")
                    
                except Exception as e:
                    logging.error(f"[{basin_id}] Ошибка при расчете PET Пенмана-Монтейта: {e}")

        # 6. Расчет динамических переменных
        try:
            logging.info(f"[{basin_id}] Расчет динамических переменных...")
            
            # t_mean_dyn (динамическое среднее значение температуры)
            if 't_mean_dyn' not in ds.variables and 'temperature_mean' in ds.variables:
                temp_mean = ds['temperature_mean'].rolling(date=30, center=True).mean()
                ds['t_mean_dyn'] = temp_mean
                ds['t_mean_dyn'].attrs = METADATA_MAP['t_mean_dyn']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная t_mean_dyn.")
            
            # p_mean_dyn (динамическое среднее количество осадков)
            if 'p_mean_dyn' not in ds.variables and 'precipitation' in ds.variables:
                precip_mean = ds['precipitation'].rolling(date=30, center=True).mean()
                ds['p_mean_dyn'] = precip_mean
                ds['p_mean_dyn'].attrs = METADATA_MAP['p_mean_dyn']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная p_mean_dyn.")
            
            # pet_mean_dyn (динамическое среднее значение PET)
            pet_var_for_dynamics = None
            if 'pet_penman_monteith' in ds.variables:
                pet_var_for_dynamics = 'pet_penman_monteith'
                logging.info(f"[{basin_id}] Используем PET Пенмана-Монтейта для динамических переменных")
            elif 'pet_priestley_taylor' in ds.variables:
                pet_var_for_dynamics = 'pet_priestley_taylor'
                logging.info(f"[{basin_id}] Используем PET Пристли-Тейлора для динамических переменных")
            
            if 'pet_mean_dyn' not in ds.variables and pet_var_for_dynamics is not None:
                pet_mean = ds[pet_var_for_dynamics].rolling(date=30, center=True).mean()
                ds['pet_mean_dyn'] = pet_mean
                ds['pet_mean_dyn'].attrs = METADATA_MAP['pet_mean_dyn']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная pet_mean_dyn (из {pet_var_for_dynamics}).")
            
            # frac_snow_dyn (доля осадков в виде снега)
            if 'frac_snow_dyn' not in ds.variables and 'temperature_mean' in ds.variables and 'precipitation' in ds.variables:
                snow_fraction = calculate_snow_fraction(
                    ds['temperature_mean'].values, 
                    ds['precipitation'].values
                )
                ds['frac_snow_dyn'] = (('date'), snow_fraction)
                ds['frac_snow_dyn'].attrs = METADATA_MAP['frac_snow_dyn']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная frac_snow_dyn.")
            
            # aridity_dyn (индекс засушливости)
            if 'aridity_dyn' not in ds.variables and pet_var_for_dynamics is not None and 'precipitation' in ds.variables:
                precip_safe = ds['precipitation'].where(ds['precipitation'] > 0.1, 0.1)
                
                pet_safe = ds[pet_var_for_dynamics].where(
                    (ds[pet_var_for_dynamics] > -10) & (ds[pet_var_for_dynamics] < 20), 
                    0.0
                )
                
                aridity = pet_safe / precip_safe
                
                aridity = aridity.where(aridity > -100, -100).where(aridity < 100, 100)
                
                ds['aridity_dyn'] = aridity
                ds['aridity_dyn'].attrs = METADATA_MAP['aridity_dyn']
                
                file_modified = True
                logging.info(f"[{basin_id}] Индекс засушливости успешно рассчитан.")
            
            # Статистики осадков
            if 'high_prec_freq_dyn' not in ds.variables and 'precipitation' in ds.variables:
                precip_stats = calculate_precipitation_statistics(ds['precipitation'], window=30)
                
                for var_name, values in precip_stats.items():
                    ds[var_name] = (('date'), values)
                    ds[var_name].attrs = METADATA_MAP[var_name]
                
                file_modified = True
                logging.info(f"[{basin_id}] Статистики осадков успешно рассчитаны.")
                
        except Exception as e:
            logging.error(f"[{basin_id}] Ошибка при расчете динамических переменных: {e}. Пропускаем.")

        # 7. Расчет лаговых переменных и скользящих сумм
        try:
            logging.info(f"[{basin_id}] Расчет лаговых переменных и скользящих сумм...")
            
            if 'precipitation_lag_1d' not in ds.variables and 'precipitation' in ds.variables:
                precip_lag_1d = ds['precipitation'].shift(date=1)
                ds['precipitation_lag_1d'] = precip_lag_1d
                ds['precipitation_lag_1d'].attrs = METADATA_MAP['precipitation_lag_1d']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная precipitation_lag_1d.")
            
            if 'precipitation_lag_2d' not in ds.variables and 'precipitation' in ds.variables:
                # Сдвигаем на 2 дня назад
                precip_lag_2d = ds['precipitation'].shift(date=2)
                ds['precipitation_lag_2d'] = precip_lag_2d
                ds['precipitation_lag_2d'].attrs = METADATA_MAP['precipitation_lag_2d']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная precipitation_lag_2d.")
            
            # Скользящие суммы осадков
            if 'precipitation_rolling_sum_7d' not in ds.variables and 'precipitation' in ds.variables:
                precip_sum_7d = ds['precipitation'].rolling(date=7, center=True).sum()
                ds['precipitation_rolling_sum_7d'] = precip_sum_7d
                ds['precipitation_rolling_sum_7d'].attrs = METADATA_MAP['precipitation_rolling_sum_7d']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная precipitation_rolling_sum_7d.")
            
            if 'precipitation_rolling_sum_30d' not in ds.variables and 'precipitation' in ds.variables:
                precip_sum_30d = ds['precipitation'].rolling(date=30, center=True).sum()
                ds['precipitation_rolling_sum_30d'] = precip_sum_30d
                ds['precipitation_rolling_sum_30d'].attrs = METADATA_MAP['precipitation_rolling_sum_30d']
                file_modified = True
                logging.info(f"[{basin_id}] Добавлена переменная precipitation_rolling_sum_30d.")
                
        except Exception as e:
            logging.error(f"[{basin_id}] Ошибка при расчете лаговых переменных: {e}. Пропускаем.")

        # 8. Принудительное обновление атрибутов для всех переменных
        try:
            logging.info(f"[{basin_id}] Обновление атрибутов переменных...")
            
            for var_name, metadata in METADATA_MAP.items():
                if var_name in ds.variables:
                    ds[var_name].attrs = metadata
                    file_modified = True
            
            logging.info(f"[{basin_id}] Атрибуты переменных обновлены.")
            
        except Exception as e:
            logging.error(f"[{basin_id}] Ошибка при обновлении атрибутов: {e}. Пропускаем.")

        # 9. Сохранение обновленного файла
        if file_modified:
            try:
                ds.to_netcdf(basin_file, mode='w')
                logging.info(f"[{basin_id}] Файл {basin_file.name} обновлен.")
            except Exception as e:
                logging.error(f"[{basin_id}] Не удалось сохранить обновленный файл {basin_file.name}: {e}.")
        
    logging.info("Обработка всех файлов завершена.")

if __name__ == "__main__":
    setup_logging()
    enrich_timeseries()