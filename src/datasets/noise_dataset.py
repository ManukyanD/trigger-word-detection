import os

import soundata
import torch
from torchaudio.transforms import Resample

from src.util.constants import SAMPLING_RATE, NOISE_DATASET_PATH


class NoiseDataset:
    def __init__(self, args):
        root = os.path.join(args.data_dir, NOISE_DATASET_PATH)
        self.dataset = soundata.initialize('urbansound8k', root)
        self.dataset.download()
        self.dataset.validate()
        self.resampler = Resample(44100, SAMPLING_RATE)
        self.sample_count = args.training_example_duration * SAMPLING_RATE

    def random(self):
        audio, sample_rate = self.dataset.choice_clip().audio
        audio = torch.from_numpy(audio)
        audio = self.resampler(audio)
        repetition_count = self.sample_count // len(audio)
        audio = audio.repeat(repetition_count)
        audio = torch.nn.functional.pad(audio, (0, self.sample_count - len(audio)))
        return audio
