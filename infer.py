import argparse
import json
import os

import torch
import torchaudio
from matplotlib import pyplot as plt
from torchaudio.transforms import Resample

from src.datasets.trigger_word_detection_dataset import TriggerWordDetectionDataset
from src.util.constants import SAMPLING_RATE
from src.util.device import to_device


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file.')

    return parser.parse_args()


def main():
    args = parse_args()

    model = torch.load(os.path.join(args.model, 'model.pt'))
    to_device(model)
    model.eval()

    with open(os.path.join(args.model, 'args.json')) as file:
        training_args = argparse.Namespace(**json.load(file))

    waveform, sampling_rate = torchaudio.load(args.audio)
    resampler = Resample(sampling_rate, SAMPLING_RATE)
    transform = TriggerWordDetectionDataset.init_transform(training_args)

    x = to_device(transform(resampler(waveform)))
    x = x[None, :]  # simulating a batch

    with torch.no_grad():
        prediction = model(x)

        plt.plot(prediction[0].cpu().numpy())
        plt.title(f'Prediction')
        plt.show()


if __name__ == '__main__':
    main()
