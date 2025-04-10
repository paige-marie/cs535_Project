import subprocess
import os
import shutil

# wget -nc -c -nd 'https://www.northwestknowledge.net/metdata/data/vpd_2024.nc'
# https://www.northwestknowledge.net/metdata/data/vpd_2024.nc

GRIDMET_VARS = ['tmmx','pr']
# GRIDMET_VARS = ['pr'] 

# YEARS = range(2013, 2023)
YEARS = range(2023, 2024)
GRIDMET_URL="https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
TMP_DIR_PATH = os.getenv("DATA_HOME")+"/data/gridmet"

def main():
    if make_downloaded_files_directory():
        for var in GRIDMET_VARS:
            for year in YEARS:
                the_download(var, year)
    else:
        print("big bad on the temp dir making")        

def the_download(var, year):
    result = subprocess.run(['wget', '-nc', '-c', '-nd', '-P', TMP_DIR_PATH, GRIDMET_URL.format(var=var, year=year)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{var} {year} BAD JOB")
        print(result.stderr)
        return False
    print(result.stdout)
    print(f'{var} {year} downloaded successful')
    return True
    
def make_downloaded_files_directory():
    # if os.path.isdir(TMP_DIR_PATH):
    #     print("deleting old data")
    #     try:
    #         shutil.rmtree(TMP_DIR_PATH)
    #     except OSError as e:
    #         print("Error: %s - %s." % (e.filename, e.strerror))
    #         return False
    
    os.makedirs(TMP_DIR_PATH, exist_ok=True)
    print(f"created {TMP_DIR_PATH}")
    return True
    
if __name__ == "__main__":
    main()
