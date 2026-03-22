"""
Модуль для преобразования скорости ветра между разными высотами.
"""

import numpy as np
from numba import njit


@njit
def convert_wind_speed_10m_to_2m(wind_speed_10m: np.ndarray, roughness_length: float = 0.12) -> np.ndarray:
    """
    Преобразует скорость ветра с высоты 10м на высоту 2м используя логарифмический профиль ветра.
    
    Согласно FAO-56, стандартная формула:
    u_2 = u_z * [4.87 / ln(67.8 * z - 5.42)]
    
    Но более точная формула с учетом шероховатости:
    u_2 = u_z * [ln((2 - d) / z_0) / ln((z - d) / z_0)]
    
    где:
    - z_0 = roughness length (длина шероховатости)
    - d = zero-plane displacement (смещение нулевой плоскости)
    - z = высота измерения (10м)
    
    Parameters
    ----------
    wind_speed_10m : np.ndarray
        Скорость ветра на высоте 10м [м/с].
    roughness_length : float, optional
        Длина шероховатости поверхности [м]. 
        По умолчанию 0.12м (травяной покров по FAO-56).
        Типичные значения:
        - Водная поверхность: 0.0002м
        - Короткая трава: 0.02м  
        - Сельхозкультуры: 0.12м (по умолчанию)
        - Лес: 1-2м
        
    Returns
    -------
    np.ndarray
        Скорость ветра на высоте 2м [м/с].
    """
    # Высоты измерения
    z1 = 10.0  # 10м
    z2 = 2.0   # 2м
    
    # Смещение нулевой плоскости (обычно d ≈ 2/3 * высота растительности)
    # Для травяного покрова принимаем d = 0
    d = 0.0
    
    # Логарифмический профиль ветра
    # u_2 = u_10 * [ln((2 - d) / z_0) / ln((10 - d) / z_0)]
    ln_term_2m = np.log((z2 - d) / roughness_length)
    ln_term_10m = np.log((z1 - d) / roughness_length)
    
    conversion_factor = ln_term_2m / ln_term_10m
    
    wind_speed_2m = wind_speed_10m * conversion_factor
    
    # Ограничиваем минимальное значение (ветер не может быть отрицательным)
    wind_speed_2m = np.maximum(wind_speed_2m, 0.0)
    
    return wind_speed_2m


@njit
def calculate_seasonal_roughness_length(doy: np.ndarray, land_use_type: str = "agricultural") -> np.ndarray:
    """
    Рассчитывает сезонный коэффициент шероховатости поверхности.
    
    Parameters
    ----------
    doy : np.ndarray
        День года [1-366].
    land_use_type : str, optional
        Тип землепользования. По умолчанию "agricultural".
        Доступные типы:
        - "agricultural": сельскохозяйственные культуры
        - "grassland": пастбища/луга
        - "forest": лес
        - "water": водная поверхность
        - "urban": городская застройка
        
    Returns
    -------
    np.ndarray
        Коэффициент шероховатости [м] для каждого дня года.
    """
    roughness = np.zeros_like(doy, dtype=np.float64)
    
    if land_use_type == "agricultural":
        # Сельскохозяйственные культуры (сезонная изменчивость)
        for i in range(len(doy)):
            day = doy[i]
            if day <= 80 or day > 330:  # Зима (янв-март, дек): голая почва/снег
                roughness[i] = 0.01
            elif day <= 150:  # Весна (апр-май): появляющаяся растительность
                # Линейное увеличение с 0.01 до 0.08
                progress = (day - 80) / 70.0
                roughness[i] = 0.01 + progress * 0.07
            elif day <= 240:  # Лето (июн-авг): зрелые культуры
                # Линейное увеличение с 0.08 до 0.20, затем стабильно
                if day <= 180:
                    progress = (day - 150) / 30.0
                    roughness[i] = 0.08 + progress * 0.12
                else:
                    roughness[i] = 0.20  # Максимальная высота культур
            elif day <= 300:  # Осень (сен-окт): уборка урожая
                # Линейное снижение с 0.20 до 0.03
                progress = (day - 240) / 60.0
                roughness[i] = 0.20 - progress * 0.17
            else:  # Поздняя осень (ноя): стерня/голая почва
                roughness[i] = 0.03
                
    elif land_use_type == "grassland":
        # Пастбища/луга (умеренная сезонная изменчивость)
        for i in range(len(doy)):
            day = doy[i]
            if day <= 90 or day > 320:  # Зима: короткая трава
                roughness[i] = 0.02
            elif day <= 180:  # Весна-лето: рост травы
                progress = (day - 90) / 90.0
                roughness[i] = 0.02 + progress * 0.08
            elif day <= 270:  # Лето: высокая трава
                roughness[i] = 0.10
            else:  # Осень: увядание
                progress = (day - 270) / 50.0
                roughness[i] = 0.10 - progress * 0.08
                
    elif land_use_type == "forest":
        # Лес (слабая сезонная изменчивость)
        for i in range(len(doy)):
            day = doy[i]
            if day <= 120 or day > 300:  # Зима: без листвы (лиственные)
                roughness[i] = 1.0
            else:  # Лето: с листвой
                roughness[i] = 1.5
                
    elif land_use_type == "water":
        # Водная поверхность (постоянная)
        roughness[:] = 0.0002
        
    elif land_use_type == "urban":
        # Городская застройка (постоянная)
        roughness[:] = 0.8
        
    else:
        # По умолчанию - сельскохозяйственный тип
        roughness[:] = 0.12
    
    return roughness


@njit 
def fao56_wind_conversion(wind_speed_10m: np.ndarray) -> np.ndarray:
    """
    Стандартная формула FAO-56 для преобразования скорости ветра с 10м на 2м.
    
    u_2 = u_z * [4.87 / ln(67.8 * z - 5.42)]
    
    Parameters
    ----------
    wind_speed_10m : np.ndarray
        Скорость ветра на высоте 10м [м/с].
        
    Returns
    -------
    np.ndarray
        Скорость ветра на высоте 2м [м/с].
    """
    # Стандартная формула FAO-56
    wind_speed_2m = wind_speed_10m * (4.87 / np.log(67.8 * 10 - 5.42))
    
    # Ограничиваем минимальное значение
    wind_speed_2m = np.maximum(wind_speed_2m, 0.0)
    
    return wind_speed_2m


@njit
def convert_wind_speed_10m_to_2m_seasonal(
    wind_speed_10m: np.ndarray, 
    doy: np.ndarray, 
    land_use_type: str = "agricultural"
) -> np.ndarray:
    """
    Преобразует скорость ветра с высоты 10м на высоту 2м с учетом сезонной изменчивости шероховатости.
    
    Parameters
    ----------
    wind_speed_10m : np.ndarray
        Скорость ветра на высоте 10м [м/с].
    doy : np.ndarray
        День года [1-366].
    land_use_type : str, optional
        Тип землепользования. По умолчанию "agricultural".
        
    Returns
    -------
    np.ndarray
        Скорость ветра на высоте 2м [м/с].
    """
    # Рассчитываем сезонную шероховатость
    roughness_seasonal = calculate_seasonal_roughness_length(doy, land_use_type)
    
    # Высоты измерения
    z1 = 10.0  # 10м
    z2 = 2.0   # 2м
    d = 0.0    # Смещение нулевой плоскости
    
    # Создаем массив для результата
    wind_speed_2m = np.zeros_like(wind_speed_10m)
    
    # Рассчитываем для каждого дня с соответствующей шероховатостью
    for i in range(len(wind_speed_10m)):
        z0 = roughness_seasonal[i]
        
        # Логарифмический профиль ветра
        ln_term_2m = np.log((z2 - d) / z0)
        ln_term_10m = np.log((z1 - d) / z0)
        
        conversion_factor = ln_term_2m / ln_term_10m
        wind_speed_2m[i] = wind_speed_10m[i] * conversion_factor
    
    # Ограничиваем минимальное значение
    wind_speed_2m = np.maximum(wind_speed_2m, 0.0)
    
    return wind_speed_2m
