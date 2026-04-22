import numpy as np
from numba import njit


@njit
def get_priestley_taylor_pet(t_min: np.ndarray, t_max: np.ndarray, s_rad: np.ndarray, lat: float, elev: float,
                             doy: np.ndarray) -> np.ndarray:
    """Рассчитывает потенциальную эвапотранспирацию (PET) по методу Пристли-Тейлора.

    Тепловой поток через почву G принимается равным 0 для суточных временных шагов (см. Newman et al., 2015 [#]_). 
    Уравнения соответствуют FAO-56 (Allen et al., 1998 [#]_).

    Параметры
    ----------
    t_min : np.ndarray
        Минимальная суточная температура [°C]
    t_max : np.ndarray
        Максимальная суточная температура [°C]
    s_rad : np.ndarray
        Солнечная радиация [Вт/м²]
    lat : float
        Широта [градусы]
    elev : float
        Высота над уровнем моря [м]
    doy : np.ndarray
        День года [1-366]

    Возвращает
    -------
    np.ndarray
        Массив PET [мм/день]
        
    Ссылки
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett, 
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale 
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional 
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration-Guidelines for computing 
        crop water requirements-FAO Irrigation and drainage paper 56. Fao, Rome, 300(9), D05109.
    """
    lat = lat * (np.pi / 180)  # градусы в радианы

    # Наклон кривой давления насыщенного пара
    t_mean = 0.5 * (t_min + t_max)
    slope_svp = _get_slope_svp_curve(t_mean)

    # Входящая чистая коротковолновая радиация
    s_rad = s_rad * 0.0864  # конвертация Вт/м² -> МДж/м²/день
    in_sw_rad = _get_net_sw_srad(s_rad)

    # Исходящая чистая длинноволновая радиация
    sol_dec = _get_sol_decl(doy)
    sha = _get_sunset_hour_angle(lat, sol_dec)
    ird = _get_ird_earth_sun(doy)
    et_rad = _get_extraterra_rad(lat, sol_dec, sha, ird)
    cs_rad = _get_clear_sky_rad(elev, et_rad)
    a_vp = _get_avp_tmin(t_min)
    out_lw_rad = _get_net_outgoing_lw_rad(t_min, t_max, s_rad, cs_rad, a_vp)

    # Чистая радиация
    net_rad = _get_net_rad(in_sw_rad, out_lw_rad)

    # Психрометрическая константа
    atm_pressure = _get_atmos_pressure(elev)
    gamma = _get_psy_const(atm_pressure)

    # PET МДж/м²/день
    alpha = 1.26  # Калиброван в CAMELS, здесь статичен
    _lambda = 2.45  # Константа, МДж/кг
    pet = (alpha / _lambda) * (slope_svp * net_rad) / (slope_svp + gamma)

    # Конверт энергию в эвапотранспирацию
    pet = pet * 0.408

    return pet


@njit
def get_priestley_taylor_pet_with_albedo(
    t_min: np.ndarray, 
    t_max: np.ndarray, 
    s_rad: np.ndarray, 
    lat: float, 
    elev: float,
    doy: np.ndarray,
    albedo: np.ndarray,
    alpha: float = 1.26
) -> np.ndarray:
    """
    Рассчитывает потенциальную эвапотранспирацию (PET) по методу Пристли-Тейлора 
    с использованием динамического альбедо.

    Параметры
    ----------
    t_min : np.ndarray
        Минимальная суточная температура [°C]
    t_max : np.ndarray
        Максимальная суточная температура [°C]
    s_rad : np.ndarray
        Солнечная радиация [Вт/м²]
    lat : float
        Широта [градусы]
    elev : float
        Высота над уровнем моря [м]
    doy : np.ndarray
        День года [1-366]
    albedo : np.ndarray
        Альбедо поверхности [-] (0-1)
    alpha : float, optional
        Коэффициент Пристли-Тейлора. По умолчанию 1.26 (калиброван в CAMELS)

    Возвращает
    -------
    np.ndarray
        Массив PET [мм/день]
    """
    lat = lat * (np.pi / 180)  # градусы в радианы

    # Наклон кривой давления насыщенного пара
    t_mean = 0.5 * (t_min + t_max)
    slope_svp = _get_slope_svp_curve(t_mean)

    # Входящая чистая коротковолновая радиация с динамическим альбедо
    s_rad = s_rad * 0.0864  # конвертация Вт/м² -> МДж/м²/день
    in_sw_rad = (1.0 - albedo) * s_rad  # Используем динамическое альбедо

    # Исходящая чистая длинноволновая радиация
    sol_dec = _get_sol_decl(doy)
    sha = _get_sunset_hour_angle(lat, sol_dec)
    ird = _get_ird_earth_sun(doy)
    et_rad = _get_extraterra_rad(lat, sol_dec, sha, ird)
    cs_rad = _get_clear_sky_rad(elev, et_rad)
    a_vp = _get_avp_tmin(t_min)
    out_lw_rad = _get_net_outgoing_lw_rad(t_min, t_max, s_rad, cs_rad, a_vp)

    # Чистая радиация
    net_rad = _get_net_rad(in_sw_rad, out_lw_rad)

    # Психрометрическая константа
    atm_pressure = _get_atmos_pressure(elev)
    gamma = _get_psy_const(atm_pressure)

    # PET МДж/м²/день
    _lambda = 2.45  # Константа, МДж/кг
    pet = (alpha / _lambda) * (slope_svp * net_rad) / (slope_svp + gamma)

    # Конверт энергию в эвапотранспирацию
    pet = pet * 0.408

    return pet


@njit
def _get_slope_svp_curve(t_mean: np.ndarray) -> np.ndarray:
    """Наклон кривой давления насыщенного пара

    Уравнение 13 FAO-56 Allen et al. (1998) 

    Параметры
    ----------
    t_mean : np.ndarray
        Средняя температура [°C] 

    Возвращает
    -------
    np.ndarray
        Наклон кривой давления насыщенного пара [кПа/°C]
    """
    delta = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / ((t_mean + 237.3)**2)
    return delta


@njit
def _get_net_sw_srad(s_rad: np.ndarray, albedo: float = 0.23) -> np.ndarray:
    """Расчет чистой коротковолновой радиации

    Уравнение 38 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    s_rad : np.ndarray
        Входящая солнечная радиация [МДж/м²/день]
    albedo : float, optional
        Альбедо, по умолчанию 0.23

    Возвращает
    -------
    np.ndarray
        Чистая коротковолновая радиация [МДж/м²/день]
    """
    net_srad = (1 - albedo) * s_rad
    return net_srad


@njit
def _get_sol_decl(doy: np.ndarray) -> np.ndarray:
    """Определение солнечного склонения

    Уравнение 24 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    doy : np.ndarray
        День года [1-366]

    Возвращает
    -------
    np.ndarray
        Солнечное склонение [радианы]
    """
    # уравнение 24 FAO Allen
    sol_dec = 0.409 * np.sin((2 * np.pi) / 365 * doy - 1.39)
    return sol_dec


@njit
def _get_sunset_hour_angle(lat: float, sol_dec: np.ndarray) -> np.ndarray:
    """Угол сумерек



    Параметры
    ----------
    lat : float
        Широта в радианах
    sol_dec : np.ndarray
        Солнечное склонение в радианах

    Возвращает
    -------
    np.ndarray
        Угол сумерек в радианах
    """
    term = -np.tan(lat) * np.tan(sol_dec)
    term[term < -1] = -1
    term[term > 1] = 1
    sha = np.arccos(term)
    return sha


@njit
def _get_ird_earth_sun(doy: np.ndarray) -> np.ndarray:
    """Обратное относительное расстояние между Землей и Солнцем

    Уравнение 23 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    doy : np.ndarray
        День года [1-366]

    Возвращает
    -------
    np.ndarray
        Обратное относительное расстояние между Землей и Солнцем
    """
    ird = 1 + 0.033 * np.cos((2 * np.pi) / 365 * doy)
    return ird


@njit
def _get_extraterra_rad(lat: float, sol_dec: np.ndarray, sha: np.ndarray, ird: np.ndarray) -> np.ndarray:
    """Внеземная радиация

    Уравнение 21 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    lat : float
        Широта в радианах
    sol_dec : np.ndarray
        Солнечное склонение в радианах
    sha : np.ndarray
        Угол сумерек в радианах
    ird : np.ndarray
        Обратное относительное расстояние между Землей и Солнцем

    Возвращает
    -------
    np.ndarray
        Внеземная радиация [МДж/м²/день]
    """
    term1 = (24 * 60) / np.pi * 0.082 * ird
    term2 = sha * np.sin(lat) * np.sin(sol_dec) + np.cos(lat) * np.cos(sol_dec) * np.sin(sha)
    et_rad = term1 * term2
    return et_rad


@njit
def _get_clear_sky_rad(elev: float, et_rad: np.ndarray) -> np.ndarray:
    """Радиация ясного неба

    Уравнение 37 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    elev : float
        Высота над уровнем моря [м]
    et_rad : np.ndarray
        Внеземная радиация [МДж/м²/день]

    Возвращает
    -------
    np.ndarray
        Радиация ясного неба [МДж/м²/день]
    """
    cs_rad = (0.75 + 2 * 10e-5 * elev) * et_rad
    return cs_rad


@njit
def _get_avp_tmin(t_min: np.ndarray) -> np.ndarray:
    """Фактическое давление пара, оцененное с использованием минимальной температуры

    Уравнение 48 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    t_min : np.ndarray
        Минимальная температура [°C]

    Возвращает
    -------
    np.ndarray
        Фактическое давление пара [кПа]
    """
    avp = 0.611 * np.exp((17.27 * t_min) / (t_min + 237.3))
    return avp


@njit
def _get_net_outgoing_lw_rad(t_min: np.ndarray, t_max: np.ndarray, s_rad: np.ndarray, cs_rad: np.ndarray,
                             a_vp: np.ndarray) -> np.ndarray:
    """Чистая исходящая длинноволновая радиация

    Температура ожидается в градусах и конвертируется в кельвины в функции.

    Уравнение 49 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    t_min : np.ndarray
        Минимальная температура [°C]
    t_max : np.ndarray
        Максимальная температура [°C]
    s_rad : np.ndarray
        Измеренная или смоделированная солнечная радиация [МДж/м²/день]
    cs_rad : np.ndarray
        Радиация ясного неба [МДж/м²/день]
    a_vp : np.ndarray
        Фактическое давление пара [кПа]

    Возвращает
    -------
    np.ndarray
        Чистая исходящая длинноволновая радиация [МДж/м²/день]
    """
    term1 = ((t_max + 273.16)**4 + (t_min + 273.16)**4) / 2  # конвертация в K в уравнении
    term2 = 0.34 - 0.14 * np.sqrt(a_vp)
    term3 = 1.35 * s_rad / cs_rad - 0.35
    stefan_boltzman = 4.903e-09
    net_lw = stefan_boltzman * term1 * term2 * term3
    return net_lw


@njit
def _get_net_rad(sw_rad: np.ndarray, lw_rad: np.ndarray) -> np.ndarray:
    """Чистая радиация

    Уравнение 40 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    sw_rad : np.ndarray
        Чистая входящая коротковолновая радиация [МДж/м²/день]
    lw_rad : np.ndarray
        Чистая исходящая длинноволновая радиация [МДж/м²/день]

    Возвращает
    -------
    np.ndarray
        [description]
    """
    return sw_rad - lw_rad


@njit
def _get_atmos_pressure(elev: float) -> float:
    """Атмосферное давление

    Уравнение 7 FAO-56 Allen et al. (1998)

    Параметры
    ----------
    elev : float
        Высота над уровнем моря [м]

    Возвращает
    -------
    float
        Атмосферное давление [кПа]
    """
    temp = (293.0 - 0.0065 * elev) / 293.0
    return np.power(temp, 5.26) * 101.3


@njit
def _get_psy_const(atm_pressure: float) -> float:
    """Психрометрическая постоянная

    Параметры
    ----------
    atm_pressure : float
        Атмосферное давление [кПа]

    Возвращает
    -------
    float
        Психрометрическая постоянная [кПа/(градус C)]
    """
    return 0.000665 * atm_pressure


@njit
def _srad_from_t(et_rad, cs_rad, t_min, t_max, coastal=False):
    """Оценка солнечной радиации по температуре"""
    # уравнение 50
    if coastal:
        adj = 0.19
    else:
        adj = 0.16

    sol_rad = adj * np.sqrt(t_max - t_min) * et_rad

    return np.minimum(sol_rad, cs_rad)
