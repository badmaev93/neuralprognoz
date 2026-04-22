import xarray as xr
import os

def view_nc_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    nc_files = [f for f in os.listdir(script_dir) if f.endswith('.nc')]
    
    print(f"Найдено файлов .nc: {len(nc_files)}")
    print(f"Папка: {script_dir}")
    
    for file in nc_files:
        file_path = os.path.join(script_dir, file)
        print(f"\nФайл: {file}")
        try:
            ds = xr.open_dataset(file_path)
            print(f"Размеры: {dict(ds.dims)}")
            print(f"Переменные: {list(ds.data_vars)}")
            ds.close()
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    view_nc_files()
