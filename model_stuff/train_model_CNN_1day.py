import socket
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
import torch.optim as optim
import json

with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/1day/hyperparamters/1day_cnn_config.json") as f:
    config = json.load(f)

EPOCHS = 20
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
SAVE_PATH = "/s/bach/b/class/cs535/cs535a/data/new_models/1day_cnn_model.pth"

## must change to correct one before running
preprocessed_dataset = "/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/1day/training/"


class ResNetNDVI(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained weights using the recommended weights syntax
        weights = R3D_18_Weights.DEFAULT
        self.base = r3d_18(weights=weights)

        # Modify the final fully connected layer for NDVI regression
        self.base.fc = nn.Linear(self.base.fc.in_features, 8 * 64 * 64)

    def forward(self, x):
        out = self.base(x)              # shape: [B, 4096]
        out = torch.tanh(out)           # NDVI values constrained to [-1, 1]
        out = out.view(-1, 8, 64, 64)   # reshape to [B, 1, 64, 64]
        return out

#def get_split_indices(dataset, split="train", test_ratio=0.2, seed=42):
#    total_indices = list(range(len(dataset)))
#    train_indices, test_indices = train_test_split(
#        total_indices, test_size=test_ratio, random_state=seed
#    )
#    return train_indices if split == "train" else test_indices


# Partition training dataset
def partition_dataset():
    dataset = MyDataset(preprocessed_dataset)
    #split_indices = get_split_indices(dataset, split="train")

    size = dist.get_world_size()
    bsz = int(BATCH_SIZE / float(size))
    part_size = len(dataset) // size
    local_indices = list(range(len(dataset)))
    local_indices = local_indices[int(dist.get_rank()) * part_size : int(dist.get_rank() + 1) * part_size]
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

    model = ResNetNDVI()

    # Modify final layer for regression (assuming single output like NDVI)


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
            #pred = pred.view(-1, 8, 64, 64)
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
            with open("/s/bach/b/class/cs535/cs535a/data/new_models/1day_CNN_model_log.txt", "a") as f:
                f.write(f"Saved {SAVE_PATH} with MSE {best_loss:.4f} at {datetime.datetime.now()}\n")



# Distributed setup

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'kale'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://kale:23456', timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    try:
        print("starting training")
        #print(os.listdir("/s/bach/b/class/cs535/cs535a/data/tiled/"))
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        setup(rank, world_size)
        print(socket.gethostname() + ": setup completed")
        train(rank, world_size)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)