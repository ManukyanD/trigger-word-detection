import math

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

from src.util.constants import *


class TriggerWordDetectionDataset(Dataset):
    def __init__(self, args):
        self.examples_path = os.path.join(args.data_dir, TRAINING_EXAMPLES_PATH)
        self.labels_path = os.path.join(args.data_dir, TRAINING_LABELS_PATH)
        self.transform = Spectrogram(n_fft=args.n_fft, win_length=args.fft_window_length,
                                     hop_length=args.fft_hop_length)
        self.hop_length = args.fft_hop_length
        self.conv_kernel_size = args.kernel_size
        self.conv_stride = args.stride

    def __getitem__(self, item):
        audio, sample_rate = torchaudio.load(os.path.join(self.examples_path, f'{item}.wav'))
        audio = self.transform(audio)
        raw_label: Tensor = torch.load(os.path.join(self.labels_path, f'{item}.pt'))
        self.label_size = math.floor((audio.shape[2] - self.conv_kernel_size) / self.conv_stride) + 1
        label = torch.zeros(self.label_size)
        non_zero_indices = raw_label.nonzero()
        non_zero_indices = non_zero_indices * self.label_size // len(raw_label)
        for index in non_zero_indices:
            end = min(index + 50, self.label_size)
            label[index: end] = 1
        return audio, label

    def __len__(self):
        return len(os.listdir(self.examples_path))
