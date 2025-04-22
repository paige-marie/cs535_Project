import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, Subset
from dataset import MyDataset  # Your dataset should output (x, y) where x is [3, T, 64, 64]

# === LSTM-based NDVI Model ===
class DeepARLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_size=512, dropout=0.2, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=64 * 64 * input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 64 * 64 * 8)  # 8-step NDVI forecast

    def forward(self, x):
        B, C, T, H, W = x.size()  # [B, 3, T, 64, 64]
        x = x.view(B, T, C * H * W)  # flatten input per timestep
        out, _ = self.lstm(x)  # [B, T, H]
        out = self.fc(out[:, -1])  # use final LSTM output
        out = torch.tanh(out)  # constrain to NDVI [-1, 1]
        return out.view(B, 8, 64, 64)  # 8-day forecast

# === RMSE ===
def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

# === Hyperparameter Grid ===
param_grid = {
    "batch_size": [500, 750],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "hidden_size": [256, 512],
    "dropout": [0.0, 0.2],
    "epochs": [5]  # keep short for tuning
}

# === Training Logic ===
def train_model(config, dataset, device):
    indices = torch.randperm(len(dataset))[:len(dataset)//4]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=config["batch_size"], shuffle=True)

    model = DeepARLSTM(
        hidden_size=config["hidden_size"],
        dropout=config["dropout"]
    ).to(device)
    criterion = nn.MSELoss()

    optimizer = {
        "adam": optim.Adam(model.parameters(), lr=config["learning_rate"]),
        "sgd": optim.SGD(model.parameters(), lr=config["learning_rate"])
    }["adam"]  # LSTMs usually perform better with Adam

    model.train()
    for epoch in range(config["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # Evaluate on the same subset
    model.eval()
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            rmse = compute_rmse(pred, y)
            total_rmse += rmse.item()
            count += 1
    return total_rmse / count

# === Main Entry Point ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to preprocessed .pt files")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--out_config", default="best_lstm_config.json")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = MyDataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_rmse = float("inf")
    best_config = None

    print("Running LSTM hyperparameter search...")
    for config in ParameterGrid(param_grid):
        print(f"Trying config: {config}")
        avg_rmse = train_model(config, dataset, device)
        print(f"   → RMSE: {avg_rmse:.4f}")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_config = config

    print("Best config:", best_config)
    print("Best RMSE:", best_rmse)

    with open(args.out_config, "w") as f:
        json.dump(best_config, f, indent=2)

if __name__ == "__main__":
    main()
