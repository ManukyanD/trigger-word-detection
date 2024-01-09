import torch.nn as nn
from torch import Tensor


class TriggerWordDetectionModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=args.fft_freq_bins, out_channels=args.filters,
                              kernel_size=args.kernel_size, stride=args.stride)
        self.batch_norm_1 = nn.BatchNorm1d(args.filters)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout1d(p=0.8)

        self.gru_1 = nn.GRU(input_size=args.filters, hidden_size=args.gru_1, batch_first=True)
        self.dropout_2 = nn.Dropout1d(p=0.8)
        self.batch_norm_2 = nn.BatchNorm1d(args.gru_1)

        self.gru_2 = nn.GRU(input_size=args.gru_1, hidden_size=args.gru_2, batch_first=True)
        self.dropout_3 = nn.Dropout1d(p=0.8)
        self.batch_norm_3 = nn.BatchNorm1d(args.gru_2)
        self.dropout_4 = nn.Dropout1d(p=0.8)

        self.linear = nn.Linear(in_features=args.gru_2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        x = x[:, 0, :, :]
        x = self.conv(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = x.permute(0, 2, 1)
        x, h_1 = self.gru_1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout_2(x)
        x = self.batch_norm_2(x)
        x = x.permute(0, 2, 1)
        x, h_2 = self.gru_2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout_3(x)
        x = self.batch_norm_3(x)
        x = self.dropout_4(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        out = self.sigmoid(x)
        out = out[:, :, 0]
        return out
