import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, Subset
from torchvision.models.video import r3d_18, R3D_18_Weights
from dataset import MyDataset

# === CNN Model Wrapper ===
class ResNetNDVI(nn.Module):
    def __init__(self):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT
        self.base = r3d_18(weights=weights)
        self.base.fc = nn.Linear(self.base.fc.in_features, 64 * 64 * 8)

    def forward(self, x):
        out = self.base(x)
        out = torch.tanh(out)
        return out.view(-1, 8, 64, 64)

# === RMSE ===
def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

# === Hyperparameter Grid ===
param_grid = {
    "batch_size": [100, 50],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "optimizer": ["adam", "sgd"],
    "epochs": [5]  # keep short for tuning
}

# === Training Function ===
def train_model(config, dataset, device):
    # Subsample 25% of data
    indices = torch.randperm(len(dataset))[:len(dataset)//4]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=config["batch_size"], shuffle=True)

    model = ResNetNDVI().to(device)
    criterion = nn.MSELoss()

    optimizer = {
        "adam": optim.Adam(model.parameters(), lr=config["learning_rate"]),
        "sgd": optim.SGD(model.parameters(), lr=config["learning_rate"])
    }[config["optimizer"]]

    model.train()
    for epoch in range(config["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # Evaluate on same subset
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

# === Main Script ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to .pt files")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--out_config", default="best_cnn_config.json")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = MyDataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_rmse = float("inf")
    best_config = None

    print("Running hyperparameter search...")
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
