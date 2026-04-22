import pandas as pd
from pathlib import Path

# --- НАСТРОЙКИ СКРИПТА ---

# Указ правильные пути к файлам
GFS_CSV_PATH = Path("../hydro_forecast_app/data/generic_dataset/raw/GFS/gfs_daily_KR_2015-04-01_to_2025-09-02.csv")
ERA5_CSV_PATH = Path("../hydro_forecast_app/data/generic_dataset/raw/ERA5/daily/era5_daily_KR_1993-01-01_to_2025-08-20.csv")

OUTPUT_DIR = Path("../hydro_forecast_app/data/generic_dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH = OUTPUT_DIR / "gfs_era5_merged_for_correction.csv"


# --- ОСНОВНАЯ ЛОГИКА ---

def main():
    print("Шаг 1: Загрузка данных...")
    try:
        df_gfs = pd.read_csv(GFS_CSV_PATH)
        df_era5 = pd.read_csv(ERA5_CSV_PATH)
        print("Данные успешно загружены.")
    except FileNotFoundError as e:
        print(f"!!! ОШИБКА: Файл не найден. Проверьте пути. {e}")
        return

    print("\nШаг 2: Обработка дат...")
    df_gfs['date'] = pd.to_datetime(df_gfs.iloc[:, 0], utc=True).dt.tz_localize(None)
    df_era5['date'] = pd.to_datetime(df_era5.iloc[:, 0], utc=True).dt.tz_localize(None)
    df_gfs = df_gfs.set_index('date')
    df_era5 = df_era5.set_index('date')
    print("Даты обработаны и установлены как индекс.")

    print("\nШаг 3: Выбор и переименование колонок...")

    rename_map = {
        'temp_mean': ['temperature_2m_above_ground_mean', 'temperature_2m_mean_daily'],
        'temp_min': ['temperature_2m_above_ground_min', 'temperature_2m_min_daily'],
        'temp_max': ['temperature_2m_above_ground_max', 'temperature_2m_max_daily'],
        'precip_sum': ['total_precipitation_surface_max', 'total_precipitation_sum_daily'],
        'wind_u': ['u_component_of_wind_10m_above_ground_mean', 'u_component_of_wind_10m_mean_daily'],
        'wind_v': ['v_component_of_wind_10m_above_ground_mean', 'v_component_of_wind_10m_mean_daily'],
        'dewpoint_mean': ['dew_point_temperature_2m_above_ground_mean', 'dewpoint_temperature_2m_mean_daily'],
        'ssrd_mean': ['downward_shortwave_radiation_flux_mean', 'surface_solar_radiation_downwards_mean_daily']
    }

    gfs_final_cols = {}
    era5_final_cols = {}

    for base_name, (gfs_col, era5_col) in rename_map.items():
        if gfs_col in df_gfs.columns:
            gfs_final_cols[gfs_col] = base_name + '_gfs'
        if era5_col in df_era5.columns:
            era5_final_cols[era5_col] = base_name + '_era5'

    print(f"Колонки для GFS: {list(gfs_final_cols.keys())}")
    print(f"Колонки для ERA5: {list(era5_final_cols.keys())}")

    df_gfs_renamed = df_gfs.rename(columns=gfs_final_cols)[list(gfs_final_cols.values())]
    df_era5_renamed = df_era5.rename(columns=era5_final_cols)[list(era5_final_cols.values())]
    print("Колонки выбраны и переименованы.")

    print("\nШаг 4: Объединение данных...")
    df_merged = df_gfs_renamed.join(df_era5_renamed, how='inner')
    
    print(f"Объединение завершено. Найдено {len(df_merged)} общих дней.")
    
    if df_merged.empty:
        print("!!! ВНИМАНИЕ: DataFrame пуст после объединения. Нет пересекающихся дат.")
        return
        
    print("\nШаг 5: Обработка пропусков...")
    print("Количество пропусков (NaN) ДО обработки:")
    print(df_merged.isnull().sum().loc[lambda x: x > 0])
    
    era5_cols = [col for col in df_merged.columns if col.endswith('_era5')]
    
    initial_rows = len(df_merged)
    df_merged.dropna(subset=era5_cols, inplace=True)
    final_rows = len(df_merged)

    print(f"Удалено {initial_rows - final_rows} строк, где отсутствовали данные реанализа (целевые переменные).")
    
    print("\nКоличество пропусков (NaN) ПОСЛЕ обработки (должно быть 0 в _era5 колонках):")
    print(df_merged.isnull().sum().loc[lambda x: x > 0])

    if df_merged.empty:
        print("!!! ВНИМАНИЕ: DataFrame пуст после удаления пропусков. Убедитесь, что временные диапазоны данных пересекаются.")
    else:
        print(f"\nШаг 6: Сохранение итогового файла в {OUTPUT_CSV_PATH}")
        df_merged.to_csv(OUTPUT_CSV_PATH)
        print("-" * 50)
        print("Скрипт успешно завершен!")
        print(f"Итоговый период данных: с {df_merged.index.min().date()} по {df_merged.index.max().date()}")
        print("Пример итоговых данных:")
        print(df_merged.head())

if __name__ == '__main__':
    main()