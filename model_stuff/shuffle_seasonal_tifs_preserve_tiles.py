import os
import shutil
import random
from collections import defaultdict

#spatially aligned data
INPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/tiled/"
OUTPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/seasonal_data/"
SEED = 42

#method for determining what season it is by day of year
def get_season(doy):
    if 60 <= doy <= 151:
        return "spring"
    elif 152 <= doy <= 243:
        return "summer"
    elif 244 <= doy <= 334:
        return "fall"
    else:
        return "winter"

#make sure all .tif files are from the same tile and day so things don't get messed up
season_groups = defaultdict(lambda: defaultdict(list))

print("Scanning .tif files")
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".tif"):
        continue
    try:
        parts = fname.split("_")
        year = int(parts[0])
        doy = int(parts[1])
        tile_id = "_".join(parts[2:])

        season = get_season(doy)
        group_key = (year, doy)

        season_groups[season][group_key].append(fname)
    except Exception as e:
        print(f"Skipping malformed filename: {fname}")


os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

#interseasonal shuffling, so Spring 2017 could be swapped with Spring 2013
print("Organizing by season")
for season, slices in season_groups.items():
    out_dir = os.path.join(OUTPUT_DIR, season)
    os.makedirs(out_dir, exist_ok=True)

    slice_keys = list(slices.keys())
    random.shuffle(slice_keys)

    for (year, doy) in slice_keys:
        for name in slices[(year, doy)]:
            src = os.path.join(INPUT_DIR, name)
            dst = os.path.join(out_dir, name)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

print("Done")
