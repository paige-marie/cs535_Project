import os
import argparse
import json
import rasterio
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_tile_id(fname):
    return "_".join(fname.split("_")[2:])  # tile_960_64.tif

def parse_year_doy(fname):
    parts = fname.split("_")
    return int(parts[0]), int(parts[1])

def get_all_tifs_recursively(root_dir):
    tif_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))
    return tif_files

def get_sorted_tile_sequences(tif_paths):
    tile_groups = defaultdict(list)
    for full_path in tif_paths:
        fname = os.path.basename(full_path)
        tile_id = get_tile_id(fname)
        tile_groups[tile_id].append(full_path)

    for tile_id in tile_groups:
        tile_groups[tile_id].sort(key=lambda path: parse_year_doy(os.path.basename(path)))

    return tile_groups

def compute_band_stats(tile_groups):
    sum_vals = np.zeros(3)
    sum_sqs = np.zeros(3)
    pixel_count = 0

    for file_list in tile_groups.values():
        for full_path in file_list:
            with rasterio.open(full_path) as src:
                temp = src.read(1).astype(np.float32)
                precip = src.read(2).astype(np.float32)
                ndvi = src.read(3).astype(np.float32)

                bands = np.stack([ndvi, precip, temp])
                flat = bands.reshape(3, -1)
                sum_vals += flat.sum(axis=1)
                sum_sqs += (flat ** 2).sum(axis=1)
                pixel_count += flat.shape[1]

    mean = sum_vals / pixel_count
    std = np.sqrt((sum_sqs / pixel_count) - mean**2)
    return mean, std

def process(tile_groups, output_dir, mean, std, lookback=8, forecast_steps=8):
    os.makedirs(output_dir, exist_ok=True)
    sample_idx = 0

    for tile_id, file_list in tqdm(tile_groups.items(), desc="Processing tiles"):
        for i in range(len(file_list) - lookback - forecast_steps + 1):
            input_paths = file_list[i:i + lookback]
            target_paths = file_list[i + lookback:i + lookback + forecast_steps]

            # Build input tensor
            input_frames = []
            for full_path in input_paths:
                with rasterio.open(full_path) as src:
                    temp = src.read(1).astype(np.float32)
                    precip = src.read(2).astype(np.float32)
                    ndvi = src.read(3).astype(np.float32)

                    bands = np.stack([ndvi, precip, temp])
                    bands = (bands - mean[:, None, None]) / std[:, None, None]
                    input_frames.append(torch.from_numpy(bands).float())

            x = torch.stack(input_frames, dim=1)  # [3, LOOKBACK, 64, 64]

            # Build target NDVI sequence
            target_frames = []
            for full_path in target_paths:
                with rasterio.open(full_path) as src:
                    ndvi = src.read(3).astype(np.float32)
                    ndvi = (ndvi - mean[0]) / std[0]  # Standardize NDVI only
                    target_frames.append(torch.tensor(ndvi, dtype=torch.float32))

            y = torch.stack(target_frames, dim=0)  # [8, 64, 64]

            torch.save((x, y), os.path.join(output_dir, f"sample_{sample_idx:05d}.pt"))
            sample_idx += 1

    print(f"Preprocessing complete: {sample_idx} samples saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Parent folder of seasonal subfolders")
    parser.add_argument("--output_dir", required=True, help="Where to store .pt files")
    parser.add_argument("--lookback", type=int, default=8, help="Number of lookback timesteps")
    parser.add_argument("--stats_dir", required=True)
    args = parser.parse_args()

    print("Collecting .tif files across all seasons...")
    all_tifs = get_all_tifs_recursively(args.input_dir)

    print("Grouping by tile and sorting by date...")
    tile_groups = get_sorted_tile_sequences(all_tifs)

    stats_path = args.stats_dir + "band_stats.json"
    if os.path.exists(stats_path):
        print("Loading existing band stats...")
        with open(stats_path, "r") as f:
            stats = json.load(f)
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
    else:
        print("Computing band stats...")
        mean, std = compute_band_stats(tile_groups)
        with open(stats_path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    print("Applying sliding windows with lookback =", args.lookback)
    process(tile_groups, args.output_dir, mean, std, lookback=args.lookback)

if __name__ == "__main__":
    main()