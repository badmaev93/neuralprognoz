"""
Модуль для расчета эталонной эвапотранспирации (ETo) по методу Пенмана-Монтейта (FAO-56).
Код адаптирован из библиотеки pyeto для работы с массивами NumPy и ускорен с помощью Numba.
"""

import numpy as np
from numba import njit

# --- Глобальные константы ---
SOLAR_CONSTANT = 0.0820 # МДж м-2 мин-1
STEFAN_BOLTZMANN_CONSTANT = 4.903e-9 # МДж K-4 м-2 день-1

@njit
def fao56_penman_monteith_eto(
    t_min: np.ndarray,
    t_max: np.ndarray,
    wind_speed_10m: np.ndarray,
    s_rad_wm2: np.ndarray,
    lat_deg: float,
    elev_m: float,
    doy: np.ndarray,
    rh_mean: np.ndarray = None,
    tdew: np.ndarray = None
) -> np.ndarray:
    """
    Рассчитывает суточную эталонную эвапотранспирацию (ETo) по методу Пенмана-Монтейта (FAO-56).
    
    Parameters
    ----------
    t_min : np.ndarray
        Минимальная суточная температура [°C].
    t_max : np.ndarray
        Максимальная суточная температура [°C].
    wind_speed_10m : np.ndarray
        Средняя суточная скорость ветра на высоте 10м [м/с].
    s_rad_wm2 : np.ndarray
        Средняя суточная солнечная радиация [Вт/м²].
    lat_deg : float
        Широта [градусы].
    elev_m : float
        Высота над уровнем моря [м].
    doy : np.ndarray
        День года [1-366].
    rh_mean : np.ndarray, optional
        Средняя суточная относительная влажность [%]. Если не указана, используется tdew.
    tdew : np.ndarray, optional
        Средняя суточная температура точки росы [°C]. Используется, если rh_mean не указана.
        Если не указаны ни rh_mean, ни tdew, влажность аппроксимируется по t_min.

    Returns
    -------
    np.ndarray
        Массив ETo [мм/день].
    """
    
    # --- Подготовительные расчеты ---
    lat_rad = lat_deg * (np.pi / 180.0) # Широта в радианах
    t_mean = (t_max + t_min) / 2.0
    
    # 1. Давление и психометрическая постоянная
    atm_pressure_kpa = 101.3 * np.power(((293.0 - 0.0065 * elev_m) / 293.0), 5.26)
    psy_const = 0.000665 * atm_pressure_kpa # гамма
    
    # 2. Давление насыщенного пара (SVP) и наклон кривой
    svp_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    svp_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    mean_svp = (svp_tmax + svp_tmin) / 2.0
    delta_svp = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / np.power((t_mean + 237.3), 2)
    
    # 3. Фактическое давление пара (AVP)
    if tdew is not None:
        avp = 0.6108 * np.exp((17.27 * tdew) / (tdew + 237.3))
    elif rh_mean is not None:
        avp = (rh_mean / 100.0) * mean_svp
    else: # Аппроксимация по t_min
        avp = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        
    # 4. Радиация
    sol_rad_mj = s_rad_wm2 * 0.0864 # из Вт/м^2 в МДж/м^2/день
    net_sw_rad = (1.0 - 0.23) * sol_rad_mj # Чистая коротковолновая радиация (альбедо = 0.23 для травы)

    # Внеземная радиация (Ra)
    sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * doy) - 1.39)
    sha_term = -np.tan(lat_rad) * np.tan(sol_dec)
    sha_term = np.clip(sha_term, -1.0, 1.0) # Ограничение для arccos
    sha = np.arccos(sha_term)
    ird = 1.0 + (0.033 * np.cos((2.0 * np.pi / 365.0) * doy))
    et_rad_term1 = (24.0 * 60.0) / np.pi * SOLAR_CONSTANT * ird
    et_rad_term2 = sha * np.sin(lat_rad) * np.sin(sol_dec) + np.cos(lat_rad) * np.cos(sol_dec) * np.sin(sha)
    et_rad = et_rad_term1 * et_rad_term2 # Ra

    # Радиация чистого неба (Rso)
    cs_rad = (0.75 + 2e-5 * elev_m) * et_rad
    cs_rad[cs_rad < 0] = 0 # Не может быть отрицательной

    # Чистая длинноволновая радиация (Rnl)
    tmax_k4 = np.power(t_max + 273.16, 4)
    tmin_k4 = np.power(t_min + 273.16, 4)
    net_lw_rad_term1 = STEFAN_BOLTZMANN_CONSTANT * ((tmax_k4 + tmin_k4) / 2.0)
    net_lw_rad_term2 = 0.34 - (0.14 * np.sqrt(avp))
    sol_rad_ratio = np.minimum(sol_rad_mj / cs_rad, 1.0) # Отношение не может быть > 1
    net_lw_rad_term3 = 1.35 * sol_rad_ratio - 0.35
    net_lw_rad = net_lw_rad_term1 * net_lw_rad_term2 * net_lw_rad_term3 # Rnl

    # 5. Чистая радиация (Rn)
    net_rad = net_sw_rad - net_lw_rad # Rn

    # 6. Скорость ветра на 2м
    wind_speed_2m = wind_speed_10m * (4.87 / np.log(67.8 * 10 - 5.42))

    # --- Уравнение Пенмана-Монтейта ---
    # Числитель
    numerator_term1 = 0.408 * delta_svp * net_rad
    numerator_term2 = psy_const * (900 / (t_mean + 273)) * wind_speed_2m * (mean_svp - avp)
    numerator = numerator_term1 + numerator_term2
    
    # Знаменатель
    denominator = delta_svp + psy_const * (1 + 0.34 * wind_speed_2m)
    
    eto = numerator / denominator
    eto[eto < 0] = 0 # ETo не может быть отрицательной
    
    return eto


@njit
def calculate_wind_speed(wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    """
    Рассчитывает скорость ветра из компонент u и v.
    
    Parameters
    ----------
    wind_u : np.ndarray
        Компонента скорости ветра по оси U (восток-запад) [м/с].
    wind_v : np.ndarray
        Компонента скорости ветра по оси V (север-юг) [м/с].
        
    Returns
    -------
    np.ndarray
        Скорость ветра [м/с].
    """
    return np.sqrt(wind_u**2 + wind_v**2)


@njit
def fao56_penman_monteith_eto_with_albedo(
    t_min: np.ndarray,
    t_max: np.ndarray,
    wind_speed_10m: np.ndarray,
    s_rad_wm2: np.ndarray,
    lat_deg: float,
    elev_m: float,
    doy: np.ndarray,
    albedo: np.ndarray,
    rh_mean: np.ndarray = None,
    tdew: np.ndarray = None,
    pressure: np.ndarray = None
) -> np.ndarray:
    """
    Рассчитывает суточную эталонную эвапотранспирацию (ETo) по методу Пенмана-Монтейта (FAO-56)
    с использованием динамического альбедо и дополнительных параметров.
    
    Parameters
    ----------
    t_min : np.ndarray
        Минимальная суточная температура [°C].
    t_max : np.ndarray
        Максимальная суточная температура [°C].
    wind_speed_10m : np.ndarray
        Средняя суточная скорость ветра на высоте 10м [м/с].
    s_rad_wm2 : np.ndarray
        Средняя суточная солнечная радиация [Вт/м²].
    lat_deg : float
        Широта [градусы].
    elev_m : float
        Высота над уровнем моря [м].
    doy : np.ndarray
        День года [1-366].
    albedo : np.ndarray
        Альбедо поверхности [-] (0-1).
    rh_mean : np.ndarray, optional
        Средняя суточная относительная влажность [%]. Если не указана, используется tdew.
    tdew : np.ndarray, optional
        Средняя суточная температура точки росы [°C]. Используется, если rh_mean не указана.
        Если не указаны ни rh_mean, ни tdew, влажность аппроксимируется по t_min.
    pressure : np.ndarray, optional
        Атмосферное давление [кПа]. Если не указано, рассчитывается по высоте.

    Returns
    -------
    np.ndarray
        Массив ETo [мм/день].
    """
    
    # --- Подготовительные расчеты ---
    lat_rad = lat_deg * (np.pi / 180.0) # Широта в радианах
    t_mean = (t_max + t_min) / 2.0
    
    # 1. Давление и психометрическая постоянная
    if pressure is not None:
        atm_pressure_kpa = pressure
    else:
        # Рассчитываем по высоте - создаем массив константных значений
        pressure_value = 101.3 * np.power(((293.0 - 0.0065 * elev_m) / 293.0), 5.26)
        atm_pressure_kpa = np.full_like(t_min, pressure_value)
    
    psy_const = 0.000665 * atm_pressure_kpa # гамма
    
    # 2. Давление насыщенного пара (SVP) и наклон кривой
    svp_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    svp_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    mean_svp = (svp_tmax + svp_tmin) / 2.0
    delta_svp = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / np.power((t_mean + 237.3), 2)
    
    # 3. Фактическое давление пара (AVP)
    if tdew is not None:
        avp = 0.6108 * np.exp((17.27 * tdew) / (tdew + 237.3))
    elif rh_mean is not None:
        avp = (rh_mean / 100.0) * mean_svp
    else: # Аппроксимация по t_min
        avp = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        
    # 4. Радиация
    sol_rad_mj = s_rad_wm2 * 0.0864 # из Вт/м^2 в МДж/м^2/день
    
    # Используем динамическое альбедо вместо фиксированного 0.23
    net_sw_rad = (1.0 - albedo) * sol_rad_mj # Чистая коротковолновая радиация

    # Внеземная радиация (Ra)
    sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * doy) - 1.39)
    sha_term = -np.tan(lat_rad) * np.tan(sol_dec)
    sha_term = np.clip(sha_term, -1.0, 1.0) # Ограничение для arccos
    sha = np.arccos(sha_term)
    ird = 1.0 + (0.033 * np.cos((2.0 * np.pi / 365.0) * doy))
    et_rad_term1 = (24.0 * 60.0) / np.pi * SOLAR_CONSTANT * ird
    et_rad_term2 = sha * np.sin(lat_rad) * np.sin(sol_dec) + np.cos(lat_rad) * np.cos(sol_dec) * np.sin(sha)
    et_rad = et_rad_term1 * et_rad_term2 # Ra

    # Радиация чистого неба (Rso)
    cs_rad = (0.75 + 2e-5 * elev_m) * et_rad
    cs_rad[cs_rad < 0] = 0 # Не может быть отрицательной

    # Чистая длинноволновая радиация (Rnl)
    tmax_k4 = np.power(t_max + 273.16, 4)
    tmin_k4 = np.power(t_min + 273.16, 4)
    net_lw_rad_term1 = STEFAN_BOLTZMANN_CONSTANT * ((tmax_k4 + tmin_k4) / 2.0)
    net_lw_rad_term2 = 0.34 - (0.14 * np.sqrt(avp))
    sol_rad_ratio = np.minimum(sol_rad_mj / cs_rad, 1.0) # Отношение не может быть > 1
    net_lw_rad_term3 = 1.35 * sol_rad_ratio - 0.35
    net_lw_rad = net_lw_rad_term1 * net_lw_rad_term2 * net_lw_rad_term3 # Rnl

    # 5. Чистая радиация (Rn)
    net_rad = net_sw_rad - net_lw_rad # Rn

    # 6. Скорость ветра на 2м
    wind_speed_2m = wind_speed_10m * (4.87 / np.log(67.8 * 10 - 5.42))

    # --- Уравнение Пенмана-Монтейта ---
    # Числитель
    numerator_term1 = 0.408 * delta_svp * net_rad
    numerator_term2 = psy_const * (900 / (t_mean + 273)) * wind_speed_2m * (mean_svp - avp)
    numerator = numerator_term1 + numerator_term2
    
    # Знаменатель
    denominator = delta_svp + psy_const * (1 + 0.34 * wind_speed_2m)
    
    eto = numerator / denominator
    eto[eto < 0] = 0 # ETo не может быть отрицательной
    
    return eto


@njit
def calculate_wind_speed(wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    """
    Рассчитывает скорость ветра из компонент u и v.
    
    Parameters
    ----------
    wind_u : np.ndarray
        Компонента скорости ветра по оси U (восток-запад) [м/с].
    wind_v : np.ndarray
        Компонента скорости ветра по оси V (север-юг) [м/с].
        
    Returns
    -------
    np.ndarray
        Скорость ветра [м/с].
    """
    return np.sqrt(wind_u**2 + wind_v**2)