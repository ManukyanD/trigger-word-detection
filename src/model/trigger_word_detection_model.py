import torch.nn as nn
from torch import Tensor
from src.util.constants import *


class TriggerWordDetectionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv1d(in_channels=CONV_IN_CHANNELS, out_channels=CONV_OUT_CHANNELS,
                              kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE)
        self.batch_norm_1 = nn.BatchNorm1d(CONV_OUT_CHANNELS)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout1d(p=0.8)

        self.gru_1 = nn.GRU(input_size=CONV_OUT_CHANNELS, hidden_size=FIRST_GRU_HIDDEN_SIZE, batch_first=True)
        self.dropout_2 = nn.Dropout1d(p=0.8)
        self.batch_norm_2 = nn.BatchNorm1d(FIRST_GRU_HIDDEN_SIZE)

        self.gru_2 = nn.GRU(input_size=FIRST_GRU_HIDDEN_SIZE, hidden_size=SECOND_GRU_HIDDEN_SIZE, batch_first=True)
        self.dropout_3 = nn.Dropout1d(p=0.8)
        self.batch_norm_3 = nn.BatchNorm1d(SECOND_GRU_HIDDEN_SIZE)
        self.dropout_4 = nn.Dropout1d(p=0.8)

        self.linear = nn.Linear(in_features=SECOND_GRU_HIDDEN_SIZE, out_features=1)
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

    def calculate_loss(self, batch):
        [x, y] = batch
        out = self(x)
        loss = nn.functional.binary_cross_entropy(out, y)
        return loss
