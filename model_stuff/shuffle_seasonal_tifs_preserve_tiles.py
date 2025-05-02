
import os
import shutil
import random
from collections import defaultdict

# === CONFIG ===
INPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/tiled/"           # Replace with your actual .tif source folder
OUTPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/seasonal_data/"    # Replace with your desired output folder
SEED = 42

# === SEASON MAPPING ===
def get_season(doy):
    if 60 <= doy <= 151:
        return "spring"
    elif 152 <= doy <= 243:
        return "summer"
    elif 244 <= doy <= 334:
        return "fall"
    else:
        return "winter"

# === PARSE FILES ===
# Organize files by: (season → list of sets of tile files for each unique year+DOY)
season_groups = defaultdict(lambda: defaultdict(list))

print("Scanning .tif files...")
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".tif"):
        continue
    try:
        parts = fname.split("_")
        year = int(parts[0])
        doy = int(parts[1])
        tile_id = "_".join(parts[2:])  # tile_960_64.tif

        season = get_season(doy)
        group_key = (year, doy)  # This represents one full tile set at one time

        season_groups[season][group_key].append(fname)
    except Exception as e:
        print(f"Skipping malformed filename: {fname}")

# === SHUFFLE AND COPY FILES ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

print("Organizing by season...")
for season, slices in season_groups.items():
    out_dir = os.path.join(OUTPUT_DIR, season)
    os.makedirs(out_dir, exist_ok=True)

    slice_keys = list(slices.keys())
    random.shuffle(slice_keys)  # Shuffle (year, doy) sets

    for (year, doy) in slice_keys:
        for fname in slices[(year, doy)]:
            src = os.path.join(INPUT_DIR, fname)
            dst = os.path.join(out_dir, fname)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

print("Done: Shuffled and organized .tif files by season with spatial sets preserved.")
