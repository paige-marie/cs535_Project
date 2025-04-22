
import torch
import torch.nn as nn

class DeepARLSTM(nn.Module):
    def __init__(self, input_channels=3, height=64, width=64, hidden_size=512, dropout=0.2, num_layers=2):
        super().__init__()

        self.input_size = input_channels * height * width  # 3 × 64 × 64 = 12288
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.height = height
        self.width = width

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, 8 * height * width)

    def forward(self, x):
        # x: [B, 3, T, 64, 64]
        B, C, T, H, W = x.shape

        # Reshape to [B, T, C×H×W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B, T, -1)       # [B, T, C×H×W]

        # LSTM forward
        lstm_out, _ = self.lstm(x)    # [B, T, hidden_size]
        last_output = lstm_out[:, -1, :]  # [B, hidden_size]

        # Predict NDVI frame (flattened), then reshape to image
        out = self.fc(last_output)        # [B, 4096]
        out = out.view(B, 8, H, W)        # [B, 1, 64, 64]

        return out
