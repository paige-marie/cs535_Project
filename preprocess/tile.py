import os
import glob
from tqdm import tqdm

import numpy as np
import rasterio
from rasterio.windows import Window #https://rasterio.readthedocs.io/en/stable/api/rasterio.windows.html

DATA = os.getenv("DATA_HOME")
INPUT_DIR = DATA + "/data/merged"
OUTPUT_DIR = DATA + "/data/tiled"
TILE_SIZE = 64

def main():
    tile_size = TILE_SIZE
    created, accepted, wrong_size, contains_nodata = 0, 0, 0, 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for input_path in tqdm(glob.glob(INPUT_DIR + "/*.tif"), ncols=0):
        # print(input_pathd)
        year, doy = extract_year_doy(input_path)
        with rasterio.open(input_path) as src:
            width = src.width
            height = src.height
            profile = src.profile
            nodata = src.nodata
            # print(f"{width=} {height=} {nodata=}") #width=1600 height=777 nodata=-32768.0
            # exit()

            for i in range(0, width, tile_size): #horozontal step
                for j in range(0, height, tile_size): #vertical step
                    w = min(tile_size, width - i)
                    h = min(tile_size, height - j)
                    created +=1

                    if w != tile_size or h != tile_size: #no partial tiles
                        wrong_size += 1
                        continue

                    window = Window(i, j, tile_size, tile_size) #(col_off, row_off, width, height) 
                    transform = src.window_transform(window)
                    tile = src.read(window=window)
                    
                    if np.any(tile == nodata): #skip tiles with nodata value
                        contains_nodata += 1
                        continue

                    accepted += 1
                    profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': transform
                    })

                    out_path = os.path.join(OUTPUT_DIR, f'{year}_{doy}_tile_{i}_{j}.tif')
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(tile)
        # break
    print(f"{created=} {accepted=} {wrong_size=} {contains_nodata=}")
    # created=46046 accepted=36432 wrong_size=9614 contains_nodata=0 
    # created=325 accepted=203 wrong_size=25 contains_nodata=97 single file, tilesize=64
                        

def extract_year_doy(path):
    filename = os.path.basename(path)
    parts = filename.split('_')
    year = int(parts[1])
    doy = int(parts[2].split('.')[0])
    return year, doy

if __name__ == "__main__":
    main()