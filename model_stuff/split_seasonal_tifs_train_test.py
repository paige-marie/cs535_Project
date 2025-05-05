import os
import shutil
import random
from collections import defaultdict


INPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/seasonal_data/"
OUTPUT_DIR = "/s/bach/b/class/cs535/cs535a/data/shuffled_data/"
SEED = 42
TRAIN_RATIO = 0.8

#parse out year and day of year
def parse_year_doy(fname):
    try:
        parts = fname.split("_")
        year = int(parts[0])
        doy = int(parts[1])
        return (year, doy)
    except:
        return None


random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#going season by season for processing
for season in ["spring", "summer", "fall", "winter"]:
    season_path = os.path.join(INPUT_DIR, season)
    if not os.path.isdir(season_path):
        print(f"Skipping missing folder: {season_path}")
        continue

    print(f"Processing season: {season}")

    #group files by year and day of year so the spatial and temporal structure is kept coherent
    slice_groups = defaultdict(list)
    for fname in sorted(os.listdir(season_path)):
        if not fname.endswith(".tif"):
            continue
        key = parse_year_doy(fname)
        if key:
            slice_groups[key].append(fname)

    #shuffle by season groups, not files, and splits into an 80/20 mix
    all_keys = list(slice_groups.keys())
    random.shuffle(all_keys)
    split_idx = int(len(all_keys) * TRAIN_RATIO)
    train_keys = set(all_keys[:split_idx])
    test_keys = set(all_keys[split_idx:])

    print(f"  Total (year, doy) blocks: {len(all_keys)}")
    print(f"  Train blocks: {len(train_keys)}")
    print(f"  Test blocks: {len(test_keys)}")

    #puts the split into the folders
    for split, key_set in [("training", train_keys), ("testing", test_keys)]:
        out_season_dir = os.path.join(OUTPUT_DIR, split, season)
        os.makedirs(out_season_dir, exist_ok=True)

        for key in key_set:
            for fname in slice_groups[key]:
                src = os.path.join(season_path, fname)
                dst = os.path.join(out_season_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

print("Done")
