
import os
import torch
import rasterio
import numpy as np

DATA_DIR = "/s/bach/b/class/cs535/cs535a/data/tiled/"
OUT_DIR = "/s/bach/b/class/cs535/cs535a/data/preprocessed_7day/"

## must be changed along with name for the different time windows we want
TIME_STEPS = 7
OFFSET = 1

os.makedirs(OUT_DIR, exist_ok=True)

all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".tif")])

for i in range(len(all_files) - TIME_STEPS - OFFSET + 1):
    sequence_paths = all_files[i : i + TIME_STEPS]
    target_path = all_files[i + TIME_STEPS + OFFSET - 1]

    frames = []
    for tif in sequence_paths:
        with rasterio.open(os.path.join(DATA_DIR, tif)) as src:
            img = src.read()  # [3, 64, 64]
            bands = np.stack([img[2] / 10000.0, img[1], img[0]])  # [NDVI, P, T]
            frames.append(torch.from_numpy(bands).float())

    x = torch.stack(frames, dim=1)  # shape: [3, T, 64, 64]

    with rasterio.open(os.path.join(DATA_DIR, target_path)) as src:
        ndvi_target = src.read(3).astype(np.float32) / 10000.0  # Band 3 = NDVI, scaled down
        y = torch.tensor(ndvi_target, dtype=torch.float32).unsqueeze(0)  # [1, 64, 64]

    torch.save((x, y), os.path.join(OUT_DIR, f"sample_{i:05d}.pt"))

    if i % 500 == 0:
        print(f"Saved {i} samples...")
