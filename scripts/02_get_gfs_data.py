import ee
import pandas as pd
from pathlib import Path
import traceback

# --- НАСТРОЙКИ СКРИПТА ---

PROJECT_ID = 'glass-scanner-356706'
AOI_ASSET_ID = 'projects/glass-scanner-356706/assets/kolyma_reservoir_basin'
BASIN_NAME = 'KR'

INITIAL_START_DATE = '2015-04-01'
END_DATE = 'latest'

TARGET_TIMEZONE = 'Asia/Magadan'

# --- Переменные по периодам ---
BASE_GFS_BANDS = [
    'temperature_2m_above_ground', 'specific_humidity_2m_above_ground',
    'relative_humidity_2m_above_ground', 'u_component_of_wind_10m_above_ground',
    'v_component_of_wind_10m_above_ground', 'precipitable_water_entire_atmosphere'
]

BANDS_2015_2017 = BASE_GFS_BANDS + ['total_precipitation_surface', 'total_cloud_cover_entire_atmosphere', 'downward_shortwave_radiation_flux']
BANDS_2018_2024 = BASE_GFS_BANDS
BANDS_2025_PLUS = BASE_GFS_BANDS + [
    'dew_point_temperature_2m_above_ground', 'u_component_of_wind_planetary_boundary_layer',
    'v_component_of_wind_planetary_boundary_layer', 'gust', 'precipitation_rate',
    'haines_index', 'ventilation_rate', 'total_cloud_cover_entire_atmosphere',
    'planetary_boundary_layer_height'
]

def get_available_bands_for_date(date_str):
    date = pd.to_datetime(date_str)
    if date.year <= 2017:
        return BANDS_2015_2017
    elif date.year >= 2025:
        return BANDS_2025_PLUS
    else:
        return BANDS_2018_2024

SCALE = 27830

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
output_dir = project_root / "data" / "generic_dataset" / "raw" / "GFS"
output_dir.mkdir(parents=True, exist_ok=True)

# --- ФУНКЦИИ ---
def find_existing_file():
    pattern = f'gfs_daily_{BASIN_NAME}_*.csv'
    matching_files = list(output_dir.glob(pattern))
    return sorted(matching_files)[-1] if matching_files else None

def generate_output_filename(start_date, end_date):
    return output_dir / f'gfs_daily_{BASIN_NAME}_{start_date}_to_{end_date}.csv'

# --- ОСНОВНАЯ ЛОГИКА ---
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
        except Exception as e:
            print(f"Не удалось прочитать файл, начинаю с {INITIAL_START_DATE}. Ошибка: {e}")
            existing_df = None
    else:
        print("Существующий файл не найден. Начинаю полную загрузку.")

    try:
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine API успешно инициализирован.")
    except Exception as e:
        print(f"Ошибка инициализации Earth Engine: {e}")
        print("Детали ошибки:")
        traceback.print_exc()
        return

    global END_DATE
    if END_DATE.lower() == 'latest':
        try:
            print("Определение последней доступной даты...")
            latest_image = ee.ImageCollection("NOAA/GFS0P25").sort('creation_time', False).first()
            END_DATE = ee.Date(latest_image.get('creation_time')).format('YYYY-MM-dd').getInfo()
            print(f"Последняя доступная дата прогноза: {END_DATE}")
        except Exception as e:
            print(f"Не удалось определить последнюю дату: {e}")
            return

    if pd.to_datetime(start_date_for_download) > pd.to_datetime(END_DATE):
        print("Данные в файле уже актуальны.")
        return

    date_ranges = pd.date_range(start=start_date_for_download, end=END_DATE, freq='D')
    all_data_list = []
    print(f"\nНачинаю пакетную загрузку с {start_date_for_download} по {END_DATE}...")

    aoi = ee.FeatureCollection(AOI_ASSET_ID)

    for i, day in enumerate(date_ranges):
        day_str = day.strftime('%Y-%m-%d')
        print(f"--- Обработка дня {i+1}/{len(date_ranges)}: {day_str} ---")
        
        try:
            start_of_day_utc = ee.Date(day_str).getRange('day').start()
            available_bands = get_available_bands_for_date(day_str)
            print(f"Доступные переменные: {len(available_bands)} шт.")

            forecast_run = (ee.ImageCollection("NOAA/GFS0P25")
                              .filter(ee.Filter.eq('creation_time', start_of_day_utc.millis()))
                              .filterDate(start_of_day_utc, start_of_day_utc.advance(1, 'day'))
                              .select(available_bands))

            # --- Обработка осадков в зависимости от периода ---
            if 'total_precipitation_surface' in available_bands:
                print("Расчет почасовых осадков из накопленных...")
                def calculate_hourly_precip(image, image_list):
                    image_list = ee.List(image_list)
                    index = image_list.indexOf(image)
                    precip_hourly = ee.Algorithms.If(
                        ee.Algorithms.IsEqual(index, 0),
                        image.select('total_precipitation_surface'),
                        image.select('total_precipitation_surface').subtract(
                            ee.Image(image_list.get(index.subtract(1))).select('total_precipitation_surface')
                        )
                    )
                    return image.addBands(ee.Image(precip_hourly).rename('total_precipitation_hourly'))
                
                forecast_list = forecast_run.toList(forecast_run.size())
                processed_collection = ee.ImageCollection.fromImages(
                    forecast_list.map(lambda img: calculate_hourly_precip(ee.Image(img), forecast_list))
                )
            else:
                processed_collection = forecast_run
            # ---кк ---

            def calculate_area_mean(image):
                date = image.date().format()
                stats = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi.geometry(), scale=SCALE, maxPixels=1e9)
                return ee.Feature(None, stats).set('date', date)

            results = processed_collection.map(calculate_area_mean).getInfo()
            if results['features']:
                all_data_list.extend([f['properties'] for f in results['features']])

        except Exception as e:
            print(f"Ошибка при обработке дня {day_str}: {e}")
            traceback.print_exc()

    if not all_data_list:
        print("Не удалось загрузить новые данные.")
        return

    print("\nДанные загружены. Начинаю агрегацию...")

    hourly_df = pd.DataFrame(all_data_list)
    hourly_df['date'] = pd.to_datetime(hourly_df['date']).dt.tz_localize('UTC').dt.tz_convert(TARGET_TIMEZONE)
    hourly_df = hourly_df.set_index('date').sort_index()
    hourly_df = hourly_df.loc[~hourly_df.index.duplicated(keep='first')]

    # ---  Динамическое создание правил агрегации ---
    base_agg_rules = {
        'temperature_2m_above_ground': ['mean', 'min', 'max'], 'specific_humidity_2m_above_ground': 'mean',
        'dew_point_temperature_2m_above_ground': 'mean', 'relative_humidity_2m_above_ground': 'mean',
        'u_component_of_wind_10m_above_ground': 'mean', 'v_component_of_wind_10m_above_ground': 'mean',
        'precipitable_water_entire_atmosphere': 'mean', 'u_component_of_wind_planetary_boundary_layer': 'mean',
        'v_component_of_wind_planetary_boundary_layer': 'mean', 'gust': 'max', 'haines_index': 'mean',
        'ventilation_rate': 'mean', 'total_cloud_cover_entire_atmosphere': 'mean',
        'planetary_boundary_layer_height': 'mean', 'downward_shortwave_radiation_flux': 'mean',
        'total_precipitation_hourly': 'sum',
        'precipitation_rate': 'sum' 
    }
    
    agg_rules_for_df = {k: v for k, v in base_agg_rules.items() if k in hourly_df.columns}
    print(f"Применяемые правила агрегации: {list(agg_rules_for_df.keys())}")
    
    daily_df_new = hourly_df.resample('D').agg(agg_rules_for_df)
    daily_df_new.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in daily_df_new.columns]
    
    # --- Приводим осадки к единому формату ---
    if 'precipitation_rate_sum' in daily_df_new.columns:
        print("Конвертация precipitation_rate в мм/день...")
        daily_df_new['precipitation_rate_sum'] *= 3600
        daily_df_new = daily_df_new.rename(columns={'precipitation_rate_sum': 'precipitation_sum_daily'})
        
    if 'total_precipitation_hourly_sum' in daily_df_new.columns:
        daily_df_new = daily_df_new.rename(columns={'total_precipitation_hourly_sum': 'precipitation_sum_daily'})
    # --- кк ---
    
    daily_df_new.dropna(how='all', inplace=True)

    if existing_df is not None:
        final_df = pd.concat([existing_df, daily_df_new])
        final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
    else:
        final_df = daily_df_new

    if final_df.empty:
        print("Итоговый датафрейм пуст.")
        return

    first_date = final_df.index.min().strftime('%Y-%m-%d')
    last_date = final_df.index.max().strftime('%Y-%m-%d')
    new_output_path = generate_output_filename(first_date, last_date)
    final_df.to_csv(new_output_path)
    
    if existing_file_path and existing_file_path != new_output_path:
        print(f"Удаляю старый файл: {existing_file_path.name}")
        existing_file_path.unlink()

    print("-" * 50)
    print(f"Скрипт завершен! Данные сохранены в: {new_output_path}")
    print(f"Всего строк: {len(final_df)}")
    print("Пример последних строк:")
    print(final_df.tail())

if __name__ == '__main__':
    main()