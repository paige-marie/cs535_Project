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
from LSTM_model import DeepARLSTM
import json
import numpy as np
import matplotlib.image as mpimg

with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/hyperparamters/4day_lstm_config.json") as f:
    config = json.load(f)

MODEL_PATH = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_lstm_model.pth"
MODEL_PATH_BIDIRECTIONAL = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_bidirectional_lstm_model.pth"
BATCH_SIZE = config["batch_size"]
TIME_STEPS = 4  # Match training window
OFFSET = 8
HIDDEN_SIZE = config["hidden_size"]
DROPOUT = config.get("dropout", 0.0)


preprocessed_dataset = "/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/test/"

stats_path = "/s/bach/b/class/cs535/cs535a/data/true_preprocessed_4day_newband_stats.json"
if os.path.exists(stats_path):
    print("Loading existing band stats...")
    with open(stats_path, "r") as f:
        stats = json.load(f)
        NDVI_mean = np.array(stats["mean"][0])
        NDVI_std = np.array(stats["std"][0])

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



def evaluate(rank, world_size, bidirectional):
    test_loader = partition_test_dataset()
    model = DeepARLSTM(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, bidirectional=bidirectional)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    if bidirectional:
        model.load_state_dict(torch.load(MODEL_PATH_BIDIRECTIONAL, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    OFFSET = 8  # Assuming 8-day forecast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_rmse_per_step = torch.zeros(OFFSET, device=device)
    total_batches = 0

    with torch.no_grad():
        for x, y in test_loader:  # y: [B, 8, 64, 64]
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            B = x.shape[0]
            pred_stack = []
            input_seq = x.clone()  # [B, 3, T, 64, 64]

            for step in range(OFFSET):
                pred = model(input_seq)  # [B, 1, 64, 64]
                pred_stack.append(pred)

                # Feedback: update NDVI channel
                input_seq[:, 0, :-1] = input_seq[:, 0, 1:]
                input_seq[:, 0, -1] = pred.squeeze(1)

            if rank == 0 and total_batches == 0:
                if bidirectional:
                    os.makedirs("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/", exist_ok=True)
                else:
                    os.makedirs("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/", exist_ok=True)

                for i in range(min(3, B)):  #Save first 3 samples
                    for t in range(OFFSET):
                        #Extract standardized prediction and ground truth
                        pred_std = pred_stack[t][i, 0].cpu().numpy()
                        target_std = y[i, t].cpu().numpy()

                        #De-standardize
                        pred_ndvi = (pred_std * NDVI_std + NDVI_mean) / 10000.0
                        target_ndvi = (target_std * NDVI_std + NDVI_mean) / 10000.0

                        #Compute RMSE in de-standardized scale
                        pixel_rmse = np.sqrt(np.mean((pred_ndvi - target_ndvi) ** 2))

                        #Save images
                        if bidirectional:
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/sample{i}_day{t+1}_pred.png", pred_ndvi, cmap='viridis', vmin=-1, vmax=1)
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/sample{i}_day{t+1}_true.png", target_ndvi, cmap='viridis', vmin=-1, vmax=1)

                            #Log RMSE for this sample
                            with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/sample_log.txt", "a") as logf:
                                logf.write(f"Sample {i} - Day {t+1}: RMSE = {pixel_rmse:.4f}\n")
                        else:
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/sample{i}_day{t+1}_pred.png", pred_ndvi, cmap='viridis', vmin=-1, vmax=1)
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/sample{i}_day{t+1}_true.png", target_ndvi, cmap='viridis', vmin=-1, vmax=1)

                            #Log RMSE for this sample
                            with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/sample_log.txt", "a") as logf:
                                logf.write(f"Sample {i} - Day {t+1}: RMSE = {pixel_rmse:.4f}\n")

            for step in range(OFFSET):
                y_step = y[:, step, :, :]
                y_pred = pred_stack[step].squeeze(1)
                rmse = compute_rmse(y_pred, y_step)
                total_rmse_per_step[step] += rmse.item()

            total_batches += 1

    avg_rmse_per_step = total_rmse_per_step / total_batches

    if rank == 0:
        for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
            print(f"Step {step}: RMSE = {rmse_val:.4f}")

        final_avg_rmse = avg_rmse_per_step.mean().item()
        print(f"\nFinal averaged RMSE across all steps: {final_avg_rmse:.4f}")

        if bidirectional:
            with open("/s/bach/b/class/cs535/cs535a/data/new_models/LSTM_bidirectional_evaluation_log_4Day.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] Final average RMSE: {final_avg_rmse:.4f}\n")
                for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
                    f.write(f"Step {step} RMSE: {rmse_val:.4f}\n")

        else:
            with open("/s/bach/b/class/cs535/cs535a/data/new_models/LSTM_evaluation_log_4Day.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] Final average RMSE: {final_avg_rmse:.4f}\n")
                for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
                    f.write(f"Step {step} RMSE: {rmse_val:.4f}\n")



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'bentley'
    os.environ['MASTER_PORT'] = '17171'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
                            init_method='tcp://bentley:23456', timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(42)


if __name__ == "__main__":
    try:
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        bidirectional = bool(sys.argv[3])
        setup(rank, world_size)
        print(os.uname()[1] + ": setup completed")
        evaluate(rank, world_size, bidirectional)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)