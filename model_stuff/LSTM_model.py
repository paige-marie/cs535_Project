
import torch
import torch.nn as nn

class DeepARLSTM(nn.Module):
    #preset hyperparameters in case I forget to add something later when it is called.
    def __init__(self, input_channels=3, height=64, width=64, hidden_size=512, dropout=0.2, num_layers=3, bidirectional=False):
        super().__init__()

        self.input_size = input_channels * height * width
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.height = height
        self.width = width

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        if bidirectional == False:
            self.fc = nn.Linear(self.hidden_size, height * width)
        if bidirectional == True:
            #doubles the output size
            self.fc = nn.Linear(self.hidden_size*2, height * width)

    def forward(self, x):
        #batch size, channels or 3 bands, time steps or lookback window, and then height and width
        B, C, T, H, W = x.shape

        #convert to a shape that LSTM likes
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, T, -1)

        #LSTM pass, only use the last time step output for the fully connected layer
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        #predict NDVI image one day forward and then reshape
        out = self.fc(last_output)        # [B, 4096]
        out = out.view(B, 1, H, W)        # [B, 1, 64, 64]

        return out
