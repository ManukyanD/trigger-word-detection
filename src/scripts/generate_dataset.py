import numpy as np
import torch
import torchaudio

from src.datasets.command_dataset import CommandDataset
from src.datasets.noise_dataset import NoiseDataset
from src.util.constants import *

project_root = os.path.abspath(os.path.join('..', '..'))


def construct_example(noise, positives, negatives):
    segments = []
    label = torch.zeros(noise.shape, dtype=torch.float32)
    example = torch.clone(noise)
    for positive in positives:
        start, end = generate_random_segment(len(noise), len(positive))
        while are_overlapping(start, end, segments):
            start, end = generate_random_segment(len(noise), len(positive))
        segments.append((start, end))
        example[start:end] = example[start:end] + positive * 5
        label[end] = 1

    for negative in negatives:
        start, end = generate_random_segment(len(noise), len(negative))
        while are_overlapping(start, end, segments):
            start, end = generate_random_segment(len(noise), len(negative))
        segments.append((start, end))
        example[start:end] = example[start:end] + negative * 5
    example /= 2
    return example, label


def generate_random_segment(interval_length, segment_length):
    start = np.random.randint(low=0, high=interval_length - segment_length)
    return start, start + segment_length


def are_overlapping(start, end, previous_segments):
    for previous in previous_segments:
        previous_start, previous_end = previous
        if start <= previous_end and end >= previous_start:
            return True
    return False


noise_dataset = NoiseDataset(os.path.join(project_root, NOISE_DATASET_PATH))
command_dataset = CommandDataset(os.path.join(project_root, COMMAND_DATASET_PATH))

examples_path = (os.path.join(project_root, TRAINING_EXAMPLES_PATH))
os.makedirs(examples_path, exist_ok=True)

labels_path = os.path.join(project_root, TRAINING_LABELS_PATH)
os.makedirs(labels_path, exist_ok=True)

print('Generating dataset')
for index in range(EXAMPLES_COUNT):
    positives_count = np.random.randint(low=0, high=MAX_POSITIVES_COUNT_IN_EXAMPLE)
    positives = [command_dataset.random_positive() for i in range(positives_count)]

    negatives_count = np.random.randint(low=0, high=MAX_NEGATIVES_COUNT_IN_EXAMPLE)
    negatives = [command_dataset.random_negative() for j in range(negatives_count)]

    noise = noise_dataset.random()

    example, y = construct_example(noise, positives, negatives)
    torchaudio.save(os.path.join(examples_path, f'{index}.wav'), example.unsqueeze(0), 16000)
    torch.save(y, os.path.join(labels_path, f'{index}.pt'))

    print(f'\r{round(index / EXAMPLES_COUNT * 100, 2)} %', end='')
