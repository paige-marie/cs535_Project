import os
import glob
import re
import numpy as np

from affine import Affine
import mercantile

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling

HOME = os.getenv("PROJECT_HOME")
GRIDMET_DIR = HOME + "/gridmet/gridmet-data"
VIIRS_DIR = HOME + "/viirs/tifs"
YEARS = range(2013, 2023)

def reproject_temperature():
    for year in YEARS:
        temp_file = glob.glob(GRIDMET_DIR+f"/tmmx_{year}.nc")[0]
        ndvi_file = glob.glob(VIIRS_DIR+f"/*NDVI*{year}*.tif")[0]
        doys = get_doys(year)

        with rasterio.open(ndvi_file) as ndvi:
            ndvi_scale = float(ndvi.tags()['scale_factor'])
            ndvi_offset = float(ndvi.tags()['add_offset'])
            ndvi_units = ndvi.tags()['units']

            with rasterio.open(temp_file) as temp:
                assert ndvi.crs == temp.crs
                temp_scale = float(temp.tags(1)['scale_factor'])
                temp_offset = float(temp.tags(1)['add_offset'])
                temp_units = temp.tags(1)['units']

                window = rasterio.windows.from_bounds(*ndvi.bounds, transform=temp.transform) #temp or ndvi?
                clipped_data = temp.read(window=window) #I think this maybe just reads the metadata, not the data itself yet
                clipped_transform = temp.window_transform(window) #affine object
                
                profile = ndvi.profile
                # https://rasterio.readthedocs.io/en/stable/topics/reproject.html
                for doy in doys:
                    data_slice = clipped_data[doy]
                    
                    resampled = np.empty((ndvi.height, ndvi.width), dtype=clipped_data.dtype)
                    profile.update({
                        "height": ndvi.height,
                        "width": ndvi.width,
                        "transform": ndvi.transform,
                        "dtype": "int16",
                        "count": 1,
                        "nodata": temp.nodata,
                    })
                    # data_slice = data_slice.astype("float32")
                    # resampled_uint16 = np.clip(np.round(resampled), 0, 65535).astype("uint16")
                    reproject(
                        source=data_slice,
                        destination=resampled,
                        src_transform=clipped_transform,
                        src_crs=temp.crs,
                        dst_transform=ndvi.transform,
                        dst_crs=ndvi.crs,
                        dst_resolution=(ndvi.transform.a, -ndvi.transform.e), #this might be ignored since I provide ndvi.transform and ndvi.crs
                        resampling=Resampling.bilinear
                    )
                    resampled_int16 = np.clip(np.round(resampled), -32768, 32767).astype("int16") # same as ndvi
                    # shared_nodata = -32768
                    # ndvi_data = np.where(ndvi_data == ndvi_nodata, shared_nodata, ndvi_data)
                    # resampled_temp = np.where(resampled_temp == temp_nodata, shared_nodata, resampled_temp)
                    with rasterio.open(f"resampled_clipped_{year}_{doy}.tif", "w", **profile) as dst:
                        dst.write(resampled_int16[np.newaxis, :, :])
                        dst.update_tags(1,
                            scale_factor=temp_scale,
                            add_offset=temp_offset,
                            units=temp_units,
                        )
        break

def omain():
    print("----------------------temp")
    with rasterio.open("/s/chopin/b/grad/pmhansen/cs535-project/cs535_Project/gridmet/gridmet-data/tmmx_2013.nc") as file:
        info(file)
    # print("----------------------pr")
    # with rasterio.open("/s/chopin/b/grad/pmhansen/cs535-project/cs535_Project/gridmet/gridmet-data/tmmx_2013.nc") as file:
    #     info(file)
    print("----------------------merged")
    with rasterio.open("merged_2013_193.tif") as file:
        info(file)
        print()
    print("----------------------viirs")
    with rasterio.open("/s/chopin/b/grad/pmhansen/cs535-project/cs535_Project/viirs/tifs/VNP13A1.001__500_m_16_days_NDVI_doy2012353_aid0001.tif") as file:
        info(file)
        print()
     
def info(dataset):
    print(f"{dataset.count=}")
    print(f"{dataset.width=}")
    print(f"{dataset.height=}")
    print(f"{dataset.crs=}")
    # print(f"{dataset.indexes=}")
    print(f"{dataset.nodata=}")
    print(f"{dataset.nodatavals=}")
    # print(f"{dataset.dtypes=}")
    print(f"{dataset.res=}")
    msk = dataset.read_masks(1)
    print(f"{dataset.dtypes[0]=}")
    print(f"{dataset.profile=}")
    print(f"{dataset.tags()}")
    print(f"{dataset.tags(1)}")
    # print(msk)
    
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
        return int(doy)
    return -1

if __name__ == "__main__":
    omain()