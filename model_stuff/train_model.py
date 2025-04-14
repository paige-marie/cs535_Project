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


# train_model_distributed.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os
import sys
import traceback
import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import MyDataset
from torchvision.models.video import r3d_18
from torchvision.models.video import r3d_18, R3D_18_Weights

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SAVE_PATH = "best_model.pth"

## must change to correct one before running
preprocessed_dataset = "/s/bach/b/class/cs535/cs535a/data/preprocessed_1day/"



def get_split_indices(dataset, split="train", test_ratio=0.2, seed=42):
    total_indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        total_indices, test_size=test_ratio, random_state=seed
    )
    return train_indices if split == "train" else test_indices


# Partition training dataset
def partition_dataset():
    dataset = MyDataset(preprocessed_dataset)
    split_indices = get_split_indices(dataset, split="train")

    size = dist.get_world_size()
    bsz = int(BATCH_SIZE / float(size))
    part_size = len(split_indices) // size
    local_indices = split_indices[int(dist.get_rank()) * part_size : int(dist.get_rank() + 1) * part_size]
    subset = Subset(dataset, local_indices)
    loader = DataLoader(subset, batch_size=bsz, shuffle=True)
    return loader


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


# Training loop

def train(rank, world_size):
    train_loader = partition_dataset()
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)

    # Modify final layer for regression (assuming single output like NDVI)
    model.fc = nn.Linear(model.fc.in_features, 64*64)

    if torch.cuda.is_available():
        model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        printProgressBar(0, len(train_loader), prefix=f"Epoch {epoch+1}", suffix='Complete', length=50)
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            pred = model(x)
            pred = pred.view(-1, 1, 64, 64)
            loss = criterion(pred, y)
            loss.backward()
            average_gradients(model)
            optimizer.step()

            epoch_loss += loss.item()
            printProgressBar(i + 1, len(train_loader), prefix=f"Epoch {epoch+1}", suffix='Complete', length=50)

        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

        if rank == 0 and (epoch_loss / len(train_loader)) < best_loss:
            best_loss = epoch_loss / len(train_loader)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved to {SAVE_PATH}")


# Distributed setup

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'juneau'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://juneau:23456', timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    try:
        print("starting training")
        print(os.listdir("/s/bach/b/class/cs535/cs535a/data/tiled/"))
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        setup(rank, world_size)
        print(socket.gethostname() + ": setup completed")
        train(rank, world_size)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)