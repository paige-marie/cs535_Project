import socket
import os
import sys
import datetime
import traceback
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import MyDataset
from LSTM_model import DeepARLSTM
import json

with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/hyperparamters/4day_lstm_config.json") as f:
    config = json.load(f)

#CONFIG
MODEL_PATH = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_lstm_model.pth"
MODEL_PATH_BIDIRECTIONAL = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_bidirectional_lstm_model.pth"
BATCH_SIZE = config["batch_size"]
TIME_STEPS = 4
LEARNING_RATE = config["learning_rate"]
HIDDEN_SIZE = config["hidden_size"]
DROPOUT = config.get("dropout", 0.0)
EPOCHS = 20
PREPROCESSED_DIR = "/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/training_new/"

#Data partitioning
#def get_split_indices(dataset, split="train", test_ratio=0.2, seed=42):
#    total_indices = list(range(len(dataset)))
#    train_indices, test_indices = train_test_split(total_indices, test_size=test_ratio, random_state=seed)
#    return train_indices if split == "train" else test_indices

def partition_dataset():
    dataset = MyDataset(preprocessed_dir=PREPROCESSED_DIR)
    #split_indices = get_split_indices(dataset, split="train")

    size = dist.get_world_size()
    bsz = int(BATCH_SIZE / float(size))
    part_size = len(dataset) // size
    local_indices = list(range(len(dataset)))
    local_indices = local_indices[int(dist.get_rank()) * part_size : int(dist.get_rank() + 1) * part_size]
    subset = Subset(dataset, local_indices)
    loader = DataLoader(subset, batch_size=bsz, shuffle=True, num_workers=4, pin_memory=True)
    return loader

#Progress bar
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

#Training loop
def train(rank, world_size, bidirectional):
    train_loader = partition_dataset()
    model = DeepARLSTM(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, bidirectional=bidirectional)
    model = model.cuda() if torch.cuda.is_available() else model
    model = nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        printProgressBar(0, len(train_loader), prefix=f"Rank {rank} Epoch {epoch}", suffix='Complete')

        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            pred = model(x)               # [B, 8, 64, 64]
            loss = criterion(pred, y[:, 0:1, :, :])    # standardized NDVI
            loss.backward()
            average_gradients(model)
            optimizer.step()

            epoch_loss += loss.item()
            printProgressBar(i + 1, len(train_loader), prefix=f"Rank {rank} Epoch {epoch}", suffix='Complete')

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nRank {rank} Epoch {epoch} Avg MSE Loss: {avg_loss:.4f}")

        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            if bidirectional:
                torch.save(model.state_dict(), MODEL_PATH_BIDIRECTIONAL)
                print(f"Model saved to {MODEL_PATH_BIDIRECTIONAL}")
            else:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Model saved to {MODEL_PATH}")
            if bidirectional:
                with open("/s/bach/b/class/cs535/cs535a/data/new_models/4day_bidirectional_LSTM_model_log.txt", "a") as f:
                    f.write(f"Saved {MODEL_PATH} with MSE {best_loss:.4f} at {datetime.datetime.now()}\n")
            else:
                with open("/s/bach/b/class/cs535/cs535a/data/new_models/4day_LSTM_model_log.txt", "a") as f:
                    f.write(f"Saved {MODEL_PATH} with MSE {best_loss:.4f} at {datetime.datetime.now()}\n")

#Distributed setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'bentley'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://bentley:23456', timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)

#Main entry point
if __name__ == "__main__":
    try:
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        bidirectional = bool(sys.argv[3])
        setup(rank, world_size)
        print(socket.gethostname() + ": setup completed")
        train(rank, world_size,bidirectional)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
