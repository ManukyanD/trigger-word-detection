import math

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

from src.util.constants import *

transform = Spectrogram(n_fft=FFT_SIZE, win_length=FFT_WIN_LENGTH, hop_length=FFT_HOP_LENGTH)


class TriggerWordDetectionDataset(Dataset):
    def __init__(self, examples_path, labels_path):
        self.examples_path = examples_path
        self.labels_path = labels_path
        spectrogram_length = math.floor(
            (EXAMPLE_DURATION * EXAMPLE_SAMPLING_RATE + 2 * FFT_HOP_LENGTH - FFT_WIN_LENGTH) / FFT_HOP_LENGTH) + 1
        self.label_size = math.floor((spectrogram_length - CONV_KERNEL_SIZE) / CONV_STRIDE) + 1

    def __getitem__(self, item):
        audio, sample_rate = torchaudio.load(os.path.join(self.examples_path, f'{item}.wav'))
        audio = transform(audio)
        raw_label: Tensor = torch.load(os.path.join(self.labels_path, f'{item}.pt'))
        label = torch.zeros(self.label_size)
        non_zero_indices = raw_label.nonzero()
        non_zero_indices = non_zero_indices * self.label_size // len(raw_label)
        for index in non_zero_indices:
            end = min(index + 50, self.label_size)
            label[index: end] = 1
        return audio, label

    def __len__(self):
        return len(os.listdir(self.examples_path))
