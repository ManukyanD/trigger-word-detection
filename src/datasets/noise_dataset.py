import soundata
import torch
from torchaudio.transforms import Resample

class NoiseDataset:
    def __init__(self, root, duration, sample_rate):
        self.dataset = soundata.initialize('urbansound8k', root)
        self.dataset.download()
        self.dataset.validate()
        self.resampler = Resample(44100, sample_rate)
        self.sample_count = duration * sample_rate

    def random(self):
        audio, sample_rate = self.dataset.choice_clip().audio
        audio = torch.from_numpy(audio)
        audio = self.resampler(audio)
        repetition_count = self.sample_count // len(audio)
        audio = audio.repeat(repetition_count)
        audio = torch.nn.functional.pad(audio, (0, self.sample_count - len(audio)))
        return audio
