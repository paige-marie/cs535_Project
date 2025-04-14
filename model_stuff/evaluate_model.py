from socket import socket
import torch
import torch.distributed as dist
import os
import sys
import numpy as np
import subprocess
import socket
import traceback
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader

# evaluate_model_distributed.py
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import traceback
import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import MyDataset
from torchvision.models.video import r3d_18

MODEL_PATH = "best_model.pth"
BATCH_SIZE = 16
TIME_STEPS = 7  # Match training window
OFFSET = 1


def get_split_indices(dataset, split="test", test_ratio=0.2, seed=42):
    total_indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        total_indices, test_size=test_ratio, random_state=seed
    )
    return train_indices if split == "train" else test_indices


def partition_test_dataset():
    dataset = MyDataset(time_steps=TIME_STEPS, offset=OFFSET)
    split_indices = get_split_indices(dataset, split="test")

    size = dist.get_world_size()
    bsz = int(BATCH_SIZE / float(size))
    part_size = len(split_indices) // size
    local_indices = split_indices[int(dist.get_rank()) * part_size : int(dist.get_rank() + 1) * part_size]
    subset = Subset(dataset, local_indices)
    loader = DataLoader(subset, batch_size=bsz, shuffle=False)
    return loader


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def evaluate(rank, world_size):
    test_loader = partition_test_dataset()
    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model)

    total_rmse = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in test_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = model(x)
            batch_rmse = compute_rmse(pred, y)
            total_rmse += batch_rmse.item()
            total_batches += 1

    avg_rmse = total_rmse / total_batches
    print(f"Rank {rank} RMSE: {avg_rmse:.4f}")

    # Gather and average RMSE across all ranks
    tensor_rmse = torch.tensor([avg_rmse], device=x.device if torch.cuda.is_available() else 'cpu')
    dist.all_reduce(tensor_rmse, op=dist.ReduceOp.SUM)
    if rank == 0:
        final_rmse = tensor_rmse.item() / world_size
        print(f"Final averaged RMSE across all ranks: {final_rmse:.4f}")

        with open("evaluation_log.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Final averaged RMSE: {final_rmse:.4f}\n")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'juneau'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://juneau:23456', timeout=datetime.timedelta(weeks=120))
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
