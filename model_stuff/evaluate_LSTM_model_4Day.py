import torch
import torch.nn as nn
import torch.distributed as dist
import os
import csv
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

#paths for normal and bidirectional lstm models
MODEL_PATH = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_lstm_model.pth"
MODEL_PATH_BIDIRECTIONAL = "/s/bach/b/class/cs535/cs535a/data/new_models/4day_bidirectional_lstm_model.pth"

TIME_STEPS = 4  # Match training window
OFFSET = 8


preprocessed_dataset = "/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/test/"

#mean and standard deviation for de-standardization
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

#partitioning for nodes
def partition_test_dataset(BATCH_SIZE):
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
def evaluate(rank, world_size, bidirectional):
    #make sure it is the right configs
    if bidirectional == "true":
        with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/hyperparamters/4day_lstm_bidirectional_config.json") as f:
            config = json.load(f)
        HIDDEN_SIZE = config["hidden_size"]
        DROPOUT = config.get("dropout", 0.0)
        BATCH_SIZE = config["batch_size"]
    if bidirectional == "false":
        with open("/s/bach/b/class/cs535/cs535a/data/shuffled_data_by_window/4day/hyperparamters/4day_lstm_config.json") as f:
            config = json.load(f)
        HIDDEN_SIZE = config["hidden_size"]
        DROPOUT = config.get("dropout", 0.0)
        BATCH_SIZE = config["batch_size"]



    test_loader = partition_test_dataset(BATCH_SIZE)
    #make sure it is the right model
    if bidirectional == "true":
        model = DeepARLSTM(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, bidirectional=True)
    if bidirectional == "false":
        model = DeepARLSTM(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, bidirectional=False)
    if torch.cuda.is_available():
        model = model.cuda()

    if bidirectional == "true":
        state_dict = torch.load(MODEL_PATH_BIDIRECTIONAL, map_location='cpu')
    if bidirectional == "false":
        state_dict = torch.load(MODEL_PATH, map_location='cpu')


    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = torch.nn.parallel.DistributedDataParallel(model)
    model.eval()

    OFFSET = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_rmse_per_step = torch.zeros(OFFSET, device=device)
    total_batches = 0

    if rank == 0:
        if bidirectional == "true":
            os.makedirs("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/", exist_ok=True)
        if bidirectional == "false":
            os.makedirs("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/", exist_ok=True)

    if bidirectional == "true":
        csv_path = "/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/LSTM_bidirectional_eval_per_sample_4day_server_" + str(rank) + ".csv"
        #setting up csv writer to write per sample results for every node
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["GlobalSampleID", "Step", "Real_NDVI_RMSE"])
    if bidirectional == "false":
        csv_path = "/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_eval_per_sample_4day_server_" + str(rank) + ".csv"

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

            #autoregressive evaluation, feeding prediction as input and saving as stack
            for step in range(OFFSET):
                pred = model(input_seq)
                pred_stack.append(pred)

                input_seq[:, 0, :-1] = input_seq[:, 0, 1:]
                input_seq[:, 0, -1] = pred.squeeze(1)

            #per sample evaluation of results and image saving
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

                    #Save images
                    if bidirectional == "true":
                        global_sample_idx = batch_idx * B + i
                        with open (csv_path, "a", newline="") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([global_sample_idx, t+1, pixel_rmse])
                        if rank == 0:
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/LSTM_bidirectional_sample{batch_idx * B + i:05d}_day{t+1}_pred_4day.png", pred_ndvi, cmap='viridis', vmin=-1, vmax=1)
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/LSTM_bidirectional_sample{batch_idx * B + i:05d}_day{t+1}_true_4day.png", target_ndvi, cmap='viridis', vmin=-1, vmax=1)

                        #Log RMSE for this sample
                        with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM_bidirectional/4Day/LSTM_bidirectional_per_sample_log_4day.txt", "a") as logf:
                            logf.write(f"Sample {i} - Day {t+1}: Real NDVI RMSE = {pixel_rmse:.4f}\n")
                    if bidirectional == "false":
                        global_sample_idx = batch_idx * B + i
                        with open (csv_path, "a", newline="") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([global_sample_idx, t+1, pixel_rmse])
                        if rank == 0:
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_sample{batch_idx * B + i:05d}_day{t+1}_pred_4day.png", pred_ndvi, cmap='viridis', vmin=-1, vmax=1)
                            mpimg.imsave(f"/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_sample{batch_idx * B + i:05d}_day{t+1}_true_4day.png", target_ndvi, cmap='viridis', vmin=-1, vmax=1)

                        #Log RMSE for this sample
                        with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_per_sample_log_4day.txt", "a") as logf:
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

    #logging stuff and printing
    if rank == 0:
        for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
            print(f"Step {step}: Real NDVI RMSE = {rmse_val:.4f}")

        final_avg_rmse = avg_rmse_per_step.mean().item()
        print(f"\nFinal averaged Real NDVI RMSE across all steps: {final_avg_rmse:.4f}")

        if bidirectional == "true":
            with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_bidirectional_evaluation_log_4Day.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] Final average Real NDVI RMSE: {final_avg_rmse:.4f}\n")
                for step, rmse_val in enumerate(avg_rmse_per_step.tolist(), 1):
                    f.write(f"Step {step} Real NDVI RMSE: {rmse_val:.4f}\n")

        if bidirectional == "false":
            with open("/s/bach/b/class/cs535/cs535a/data/eval_results/LSTM/4Day/LSTM_evaluation_log_4Day.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] Final average Real NDVI RMSE: {final_avg_rmse:.4f}\n")
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
        bidirectional = sys.argv[3].lower()
        setup(rank, world_size)
        print(os.uname()[1] + ": setup completed")
        evaluate(rank, world_size, bidirectional)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)