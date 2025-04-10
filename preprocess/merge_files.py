import os
import glob
import re
import numpy as np
from pprint import pprint
from tqdm import tqdm

from affine import Affine
import mercantile

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling

HOME = os.getenv("PROJECT_HOME")
DATA = os.getenv("DATA_HOME")
GRIDMET_DIR = DATA + "/data/gridmet"
VIIRS_DIR = DATA + "/data/viirs"
OUTPUT_DIR = DATA + "/data/merged"
YEARS = range(2013, 2024)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # open temp
    # open pr
    # get correct viirs file
    for year in tqdm(YEARS, ncols=0):
        temp_file = glob.glob(GRIDMET_DIR+f"/tmmx_{year}.nc")[0]
        pr_file = glob.glob(GRIDMET_DIR+f"/pr_{year}.nc")[0]
        doys = get_doys(year)
        merge_with_gridmet(year, temp_file, pr_file, doys)
        # break
        
def merge_with_gridmet(year: int, temp_file: str, pr_file:str, doys:list[str]):
    with rasterio.open(temp_file) as temp:
        with rasterio.open(pr_file) as pr:
            for doy in doys:
                ndvi_file = glob.glob(VIIRS_DIR+f"/*NDVI*{year}{doy}*.tif")[0]
                with rasterio.open(ndvi_file) as ndvi:
                    process_and_write(temp, pr, ndvi, year, doy)
                # break

def process_and_write(temp, pr, ndvi, year, doy):
    doy = int(doy)
    assert ndvi.crs == temp.crs and ndvi.crs == pr.crs
    
    profile = ndvi.profile
    
    NODATA_VAL = -32768
    
    profile.update({
        "height": ndvi.height,
        "width": ndvi.width,
        "transform": ndvi.transform,
        "dtype": "int16",
        "count": 3,
        "nodata": NODATA_VAL,
    })
    window = rasterio.windows.from_bounds(*ndvi.bounds, transform=temp.transform) #temp or ndvi?
    temp_clipped = temp.read(window=window)
    temp_transform = temp.window_transform(window) #affine object
    
    pr_clipped = pr.read(window=window)
    pr_transform = pr.window_transform(window) #affine object
    
    temp_slice = temp_clipped[doy]
    pr_slice = pr_clipped[doy]
    
    temp_slice = np.where(temp_slice == 32767, NODATA_VAL, temp_slice)
    pr_slice = np.where(pr_slice == 32767, NODATA_VAL, pr_slice)
    ndvi_data = ndvi.read(1)
    ndvi_data = np.where(ndvi_data == -15000, NODATA_VAL, ndvi_data)
    
    temp_resampled = np.empty((ndvi.height, ndvi.width), dtype=temp_clipped.dtype)
    pr_resampled = np.empty((ndvi.height, ndvi.width), dtype=pr_clipped.dtype)
    
    reproject(
        source=temp_slice, destination=temp_resampled,
        src_transform=temp_transform, src_crs=temp.crs,
        dst_transform=ndvi.transform, dst_crs=ndvi.crs,
        # dst_resolution=(ndvi.transform.a, -ndvi.transform.e), #this might be ignored since I provide ndvi.transform and ndvi.crs
        resampling=Resampling.bilinear
    )
    
    reproject(
        source=pr_slice, destination=pr_resampled,
        src_transform=pr_transform, src_crs=temp.crs,
        dst_transform=ndvi.transform, dst_crs=ndvi.crs,
        # dst_resolution=(ndvi.transform.a, -ndvi.transform.e), #this might be ignored since I provide ndvi.transform and ndvi.crs
        resampling=Resampling.bilinear
    )
    
    temp_resampled_int16 = np.clip(np.round(temp_resampled), -32768, 32767).astype("int16") # same as ndvi
    pr_resampled_int16 = np.clip(np.round(pr_resampled), -32768, 32767).astype("int16") # same as ndvi
    
    with rasterio.open(f"{OUTPUT_DIR}/merged_{year}_{doy}.tif", "w", **profile) as dst:
        dst.write(temp_resampled_int16, 1)
        dst.update_tags(1, **grab_gridmet_tags(temp))
        dst.write(pr_resampled_int16, 2)
        dst.update_tags(2, **grab_gridmet_tags(pr))
        dst.write(ndvi_data, 3)
        dst.update_tags(3, **grab_ndvi_tags(ndvi))

def grab_gridmet_tags(dataset):
    return {
        "scale_factor" : float(dataset.tags(1)['scale_factor']),
        "add_offset" : float(dataset.tags(1)['add_offset']),
        "units" : dataset.tags(1)['units']
    }
    
def grab_ndvi_tags(ndvi):
    return {
        "scale_factor" : float(ndvi.tags()['scale_factor']),
        "add_offset" : float(ndvi.tags()['add_offset']),
        "units" : ndvi.tags()['units']
    }
    
def get_doys(year):
    doys = list()
    for file in glob.glob(VIIRS_DIR+f"/*NDVI*{year}*.tif"):
        doy = get_year_doy(file)
        if doy != -1:
            doys.append(doy)
    return doys

def get_year_doy(filename):
    match = re.search(r'doy(\d{4})(\d{3})', filename)
    if match:
        year = match.group(1)
        doy = match.group(2)
        return doy
        # return int(doy)
    return -1


if __name__ == "__main__":
    main()