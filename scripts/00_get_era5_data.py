import ee
import pandas as pd
from pathlib import Path
import time

# --- НАСТРОЙКИ СКРИПТА ---

PROJECT_ID = '' #ВВЕСТИ
AOI_ASSET_ID = f'projects/{PROJECT_ID}/assets/kolyma_reservoir_basin' #ШЕЙП ФАЙЛ ВОДОСБОРНОГО БАССЕЙНА
BASIN_NAME = 'KR'  # Название бассейна для имени файла

# Начальная дата для самой первой загрузки. 
INITIAL_START_DATE = '1993-01-01' 
END_DATE = 'latest'

TARGET_TIMEZONE = 'Asia/Magadan' # GMT+11

HOURLY_BANDS = [
    'total_precipitation_hourly', 'temperature_2m', 'surface_solar_radiation_downwards',
    'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure',
    'forecast_albedo', 'dewpoint_temperature_2m', 'snow_depth_water_equivalent',
    'snowfall_hourly', 'snowmelt_hourly',
]

SCALE = 27830

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
output_dir = project_root / "data" / "generic_dataset" / "raw" / "ERA5" / "daily"
output_dir.mkdir(parents=True, exist_ok=True)

def find_existing_file():
    """Ищет существующий файл с данными для данного бассейна"""
    pattern = f'era5_daily_{BASIN_NAME}_*.csv'
    matching_files = list(output_dir.glob(pattern))
    if matching_files:
        return sorted(matching_files)[-1]
    return None

def generate_output_filename(start_date, end_date):
    """Генерирует имя файла на основе дат"""
    return output_dir / f'era5_daily_{BASIN_NAME}_{start_date}_to_{end_date}.csv'

# --- КОНЕЦ НАСТРОЕК ---

def main():    
    start_date_for_download = INITIAL_START_DATE
    existing_df = None
    existing_file_path = find_existing_file()

    if existing_file_path:
        print(f"Обнаружен существующий файл: {existing_file_path}")
        try:
            existing_df = pd.read_csv(existing_file_path, index_col='date', parse_dates=True)
            if not existing_df.empty:
                last_date = existing_df.index.max()
                start_date_for_download = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"Последняя дата в файле: {last_date.date()}. Начинаю загрузку с {start_date_for_download}.")
            else:
                print("Файл пуст. Начинаю загрузку с начальной даты.")
                existing_df = None # Считаем, что его нет, если он пуст
        except Exception as e:
            print(f"Не удалось прочитать существующий файл: {e}. Начинаю загрузку с начальной даты.")
            existing_df = None
    else:
        print("Существующий файл не найден. Начинаю полную загрузку.")



    try:
        ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
        print("Earth Engine API успешно инициализирован.")
    except Exception as e:
        print(f"Ошибка инициализации Earth Engine: {e}")
        return

    global END_DATE
    if END_DATE.lower() == 'latest':
        try:
            print("Определение последней доступной даты...")
            collection_for_date = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            latest_image = collection_for_date.sort('system:time_start', False).first()
            end_date_str = ee.Date(latest_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            END_DATE = end_date_str
            print(f"Последняя доступная дата: {END_DATE}")
        except Exception as e:
            print(f"Не удалось определить последнюю дату: {e}")
            return

    # ---  ПРОВЕРКА, НУЖНО ЛИ ВООБЩЕ ЧТО-ТО ГРУЗИТЬ ---
    if pd.to_datetime(start_date_for_download) > pd.to_datetime(END_DATE):
        print("Данные в файле уже актуальны. Загрузка не требуется.")
        
        if existing_df is not None and existing_file_path:
            first_date = existing_df.index.min().strftime('%Y-%m-%d')
            last_date = existing_df.index.max().strftime('%Y-%m-%d')
            new_output_path = generate_output_filename(first_date, last_date)
            
            if existing_file_path != new_output_path:
                print(f"Переименовываю файл с актуальными датами...")
                print(f"Старое имя: {existing_file_path.name}")
                print(f"Новое имя: {new_output_path.name}")
                existing_file_path.rename(new_output_path)
        
        print("-" * 50)
        print(f"Скрипт успешно завершен!")
        return
    # --- кк ---

    # Для коротких периодов (менее месяца) обрабат весь период сразу
    start_dt = pd.to_datetime(start_date_for_download)
    end_dt = pd.to_datetime(END_DATE)
    
    if (end_dt - start_dt).days <= 31:
        date_ranges = [start_dt]
    else:
        # Длинный период - разб по месяцам
        date_ranges = pd.date_range(start=start_date_for_download, end=END_DATE, freq='MS')
    
    all_data_list = []
    print(f"\nНачинаю пакетную загрузку ПОЧАСОВЫХ данных с {start_date_for_download} по {END_DATE}...")
    print(f"Количество батчей для обработки: {len(date_ranges)}")
    print(f"Диапазоны дат: {date_ranges}")

    for i, start_chunk in enumerate(date_ranges):
        if isinstance(start_chunk, pd.Timestamp):
            end_chunk = end_dt
        else:
            end_chunk = start_chunk + pd.offsets.MonthEnd(1)
            if end_chunk > pd.to_datetime(END_DATE):
                end_chunk = pd.to_datetime(END_DATE)
        start_str, end_str = start_chunk.strftime('%Y-%m-%d'), end_chunk.strftime('%Y-%m-%d')
        print(f"--- Загрузка батча {i+1}/{len(date_ranges)}: с {start_str} по {end_str} ---")
        try:
            aoi = ee.FeatureCollection(AOI_ASSET_ID)
            filtered_collection = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                                   .filterDate(start_str, end_str)
                                   .select(HOURLY_BANDS)
                                   .filterBounds(aoi.geometry()))
            count = filtered_collection.size().getInfo()
            print(f"Найдено изображений: {count}")
            if count == 0:
                print("Изображений нет, пропускаю.")
                continue
            def calculate_area_mean(image):
                date = image.date().format()
                stats = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi.geometry(), scale=SCALE, maxPixels=1e9)
                return ee.Feature(None, stats).set('date', date)
            processed_fc = filtered_collection.map(calculate_area_mean)
            results = processed_fc.getInfo()
            if results['features']:
                chunk_data = [f['properties'] for f in results['features']]
                all_data_list.extend(chunk_data)
        except Exception as e:
            print(f"Ошибка GEE при обработке батча {start_str} - {end_str}: {e}")
            import traceback
            traceback.print_exc()

    if not all_data_list:
        print("Не удалось загрузить новые почасовые данные. Файл не будет изменен.")
        return

    print("\nНовые почасовые данные успешно загружены. Начинаю агрегацию...")

    hourly_df = pd.DataFrame(all_data_list)
    hourly_df.dropna(how='all', inplace=True)
    hourly_df['date'] = pd.to_datetime(hourly_df['date']).dt.tz_localize('UTC').dt.tz_convert(TARGET_TIMEZONE)
    hourly_df = hourly_df.drop_duplicates(subset='date').set_index('date').sort_index()

    agg_rules = {band: 'mean' for band in HOURLY_BANDS}
    agg_rules['total_precipitation_hourly'] = 'sum'
    agg_rules['snowfall_hourly'] = 'sum'
    agg_rules['snowmelt_hourly'] = 'sum'
    agg_rules['temperature_2m'] = ['mean', 'min', 'max']

    print(f"Агрегация в суточные данные для часового пояса {TARGET_TIMEZONE}...")
    daily_df_new = hourly_df.resample('D').agg(agg_rules)
    
    daily_df_new.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in daily_df_new.columns]
    daily_df_new.dropna(how='all', inplace=True)
    
    rename_map = {
        col: f"{col.replace('_hourly', '')}_daily" for col in daily_df_new.columns
    }
    daily_df_new = daily_df_new.rename(columns=rename_map)

    # ---  ОБЪЕДИНЕНИЕ И СОХРАНЕНИЕ ---
    if existing_df is not None:
        print("Объединяю новые данные с существующими...")
        final_df = pd.concat([existing_df, daily_df_new])
        final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
    else:
        final_df = daily_df_new

    first_date = final_df.index.min().strftime('%Y-%m-%d')
    last_date = final_df.index.max().strftime('%Y-%m-%d')
    new_output_path = generate_output_filename(first_date, last_date)
    
    final_df.to_csv(new_output_path)
    
    if existing_file_path and existing_file_path != new_output_path:
        print(f"Удаляю старый файл: {existing_file_path}")
        existing_file_path.unlink()
    
    # --- кк ---

    print("-" * 50)
    print(f"Скрипт успешно завершен!")
    print(f"СУТОЧНЫЕ данные сохранены/обновлены в файле: {str(new_output_path)}")
    print(f"Всего строк (суток) в итоговом файле: {len(final_df)}")
    print("Пример последних 5 строк суточных данных:")
    temp_cols = [col for col in final_df.columns if 'temperature' in col]
    print(final_df[temp_cols].tail())


if __name__ == '__main__':
    main()