import torch
import torch.nn as nn
import torch.distributed as dist
import os
import csv
import sys
import traceback
import datetime

from click.core import batch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import MyDataset
from torchvision.models.video import r3d_18
from torchvision.models.video import r3d_18, R3D_18_Weights
import json
import numpy as np
import matplotlib as plt
import matplotlib.image as mpimg

#grab configs
with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/1day/hyperparamters/1day_cnn_config.json") as f:
    config = json.load(f)

MODEL_PATH = "/s/bach/b/class/cs535/cs535a/data/new_models/1day_cnn_model.pth"
BATCH_SIZE = config["batch_size"]
TIME_STEPS = 1
OFFSET = 8


preprocessed_dataset = "/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/1day/test/"

#grab mean and standard deviation for later de-standardizing
stats_path = "/s/bach/b/class/cs535/cs535a/data/true_preprocessed_1day_new_band_stats.json"
if os.path.exists(stats_path):
    print("Loading existing band stats...")
    with open(stats_path, "r") as f:
        stats = json.load(f)
        NDVI_mean = np.array(stats["mean"][0])
        NDVI_std = np.array(stats["std"][0])


class ResNetNDVI(nn.Module):
    def __init__(self):
        super().__init__()

        #load pretrained weights
        weights = R3D_18_Weights.DEFAULT
        self.base = r3d_18(weights=weights)

        #modify final layer
        self.base.fc = nn.Linear(self.base.fc.in_features, 64 * 64)

    def forward(self, x):
        out = self.base(x)
        out = torch.tanh(out)
        out = out.view(-1, 1, 64, 64)
        return out

#def get_split_indices(dataset, split="test", test_ratio=0.2, seed=42):
#    total_indices = list(range(len(dataset)))
#    train_indices, test_indices = train_test_split(
#        total_indices, test_size=test_ratio, random_state=seed
#    )
#    return train_indices if split == "train" else test_indices


def partition_test_dataset():
    dataset = MyDataset(preprocessed_dataset)
    #split_indices = get_split_indices(dataset, split="test")

    size = dist.get_world_size()
    bsz = int(BATCH_SIZE / float(size))
    part_size = len(dataset) // size
    local_indices = list(range(len(dataset)))
    local_indices = local_indices[int(dist.get_rank()) * part_size : int(dist.get_rank() + 1) * part_size]
    subset = Subset(dataset, local_indices)
    loader = DataLoader(subset, batch_size=bsz, shuffle=False)
    return loader


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

#eval
def evaluate(rank, world_size):
    test_loader = partition_test_dataset()
    model = ResNetNDVI()
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    total_rmse_per_step = torch.zeros(OFFSET, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    total_batches = 0

    if rank == 0:
        os.makedirs("/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/", exist_ok=True)

    csv_path = "/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/CNN_eval_per_sample_1day_server_" + str(rank) + ".csv"
    #setup csv writer for per sample logging
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["GlobalSampleID", "Step", "Real_NDVI_RMSE"])

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            B = x.shape[0]
            pred_stack = []
            input_seq = x.clone()

            #autoregressive loop, predict an NDVI image and feed back as input, then stack
            for step in range(OFFSET):
                pred = model(input_seq)
                pred_stack.append(pred)

                input_seq[:, 0, :-1] = input_seq[:, 0, 1:]
                input_seq[:, 0, -1] = pred.squeeze(1)

            #loop for assessing autoregressive prediction and image generation on master node only
            for i in range(B):
                for t in range(OFFSET):
                    #extract standardized prediction and ground truth
                    pred_std = pred_stack[t][i, 0].cpu().numpy()
                    target_std = y[i, t].cpu().numpy()

                    #De-standardize
                    pred_ndvi = (pred_std * NDVI_std + NDVI_mean) / 10000.0
                    target_ndvi = (target_std * NDVI_std + NDVI_mean) / 10000.0

                    #Compute RMSE in de-standardized scale
                    pixel_rmse = np.sqrt(np.mean((pred_ndvi - target_ndvi) ** 2))

                    #id for Jackson to use later
                    global_sample_idx = batch_idx * B + i
                    with open (csv_path, "a", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([global_sample_idx, t+1, pixel_rmse])

                    #Save images
                    if rank == 0:
                        mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/CNN_sample{batch_idx * B + i:05d}_day{t+1}_pred_1day.png", pred_ndvi, cmap='viridis', vmin=-1, vmax=1)
                        mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/CNN_sample{batch_idx * B + i:05d}_day{t+1}_true_1day.png", target_ndvi, cmap='viridis', vmin=-1, vmax=1)

                    #Log RMSE for this sample
                    with open("/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/CNN_per_sample_log_1day.txt", "a") as logf:
                        logf.write(f"Sample {i} - Day {t+1}: Real NDVI RMSE = {pixel_rmse:.4f}\n")

            #per step RMSE averaging
            for step in range(OFFSET):
                y_step_std = y[:, step, :, :]
                y_pred_std = pred_stack[step].squeeze(1)

                y_step = (y_step_std.cpu() * NDVI_std + NDVI_mean) / 10000.0
                y_pred = (y_pred_std.cpu() * NDVI_std + NDVI_mean) / 10000.0

                rmse = compute_rmse(y_pred, y_step)
                total_rmse_per_step[step] += rmse.item()

            total_batches += 1

    avg_rmse_per_step = total_rmse_per_step / total_batches

#printing and logging stuff
    if rank == 0:
        for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
            print(f"Step {step}: Real NDVI RMSE = {rmse_val:.4f}")

        final_avg_rmse = avg_rmse_per_step.mean().item()
        print(f"\nFinal averaged Real NDVI RMSE across all steps: {final_avg_rmse:.4f}")

        with open("/s/bach/b/class/cs535/cs535a/data/eval_results/CNN/1Day/CNN_evaluation_log_1Day.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Final average  Real NDVI RMSE: {final_avg_rmse:.4f}\n")
            for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
                f.write(f"Step {step} Real NDVI RMSE: {rmse_val:.4f}\n")

#distributed set up
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'mars'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://mars:23456', timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    try:
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        setup(rank, world_size)
        print(os.uname()[1] + ": setup completed")
        evaluate(rank, world_size)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
