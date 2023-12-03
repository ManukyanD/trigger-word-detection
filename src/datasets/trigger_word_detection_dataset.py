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

    def __getitem__(self, item):
        audio, sample_rate = torchaudio.load(os.path.join(self.examples_path, f'{item}.wav'))
        audio = transform(audio)
        raw_label: Tensor = torch.load(os.path.join(self.labels_path, f'{item}.pt'))
        label_size = self.calculate_label_size(len(raw_label))
        label = torch.zeros(label_size)
        non_zero_indices = raw_label.nonzero()
        non_zero_indices = non_zero_indices * label_size // len(raw_label)
        for index in non_zero_indices:
            end = min(index + 50, label_size)
            label[index: end] = 1
        return audio, label

    def __len__(self):
        return len(os.listdir(self.examples_path))

    def calculate_label_size(self, raw_label_length):
        spectrogram_length = (raw_label_length - FFT_WIN_LENGTH) / FFT_HOP_LENGTH + 1
        return int((spectrogram_length - CONV_KERNEL_SIZE) / CONV_STRIDE + 1)
