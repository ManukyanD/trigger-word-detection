import os.path
import random
from pathlib import Path

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from src.util.constants import COMMANDS_DATASET_PATH


class CommandDataset(SPEECHCOMMANDS):
    def __init__(self, args):
        root = os.path.join(args.data_dir, COMMANDS_DATASET_PATH)
        os.makedirs(root, exist_ok=True)
        super().__init__(root, download=True)
        base_dir = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
        all_words = [dirname for dirname in os.listdir(base_dir) if
                     os.path.isdir(os.path.join(base_dir, dirname)) and dirname != '_background_noise_']
        if args.trigger_word not in all_words:
            raise ValueError(
                f'Unknown trigger word "{args.trigger_word}" was provided. Trigger word must be one of {all_words}.')
        self.positives = [str(path) for path in Path(os.path.join(base_dir, args.trigger_word)).glob('*.wav')]
        negative_words = [word for word in all_words if word != args.trigger_word]
        self.negatives = [str(path) for neg in negative_words for path in
                          Path(os.path.join(base_dir, neg)).glob('*.wav')]

    def random_negative(self):
        audio, sample_rate = torchaudio.load(random.choice(self.negatives))
        return audio[0]

    def random_positive(self):
        audio, sample_rate = torchaudio.load(random.choice(self.positives))
        return audio[0]
