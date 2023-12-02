import os.path
from pathlib import Path

import numpy as np
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample


class CommandDataset(SPEECHCOMMANDS):
    def __init__(self, root, trigger_word, sample_rate):
        os.makedirs(root, exist_ok=True)
        super().__init__(root, download=True)
        root = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
        all_files_list = sorted(str(path) for path in Path(root).glob('*/*.wav'))
        example_files_list = [path for path in all_files_list if
                              '_nohash_' in path and '_background_noise_' not in path]
        positive_path_prefix = os.path.join(root, trigger_word)
        self.positives = [path for path in example_files_list if path.startswith(positive_path_prefix)]
        self.negatives = [path for path in example_files_list if not path.startswith(positive_path_prefix)]
        self.resampler = Resample(16000, sample_rate)

    def random_negative(self):
        audio, sample_rate = torchaudio.load(self.negatives[np.random.randint(low=0, high=len(self.negatives))])
        return self.resampler(audio[0])

    def random_positive(self):
        audio, sample_rate = torchaudio.load(self.positives[np.random.randint(low=0, high=len(self.positives))])
        return self.resampler(audio[0])
