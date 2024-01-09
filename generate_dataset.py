import argparse
import os

import numpy as np
import torch
import torchaudio

from src.datasets.command_dataset import CommandDataset
from src.datasets.noise_dataset import NoiseDataset
from src.util.constants import TRAINING_EXAMPLES_PATH, TRAINING_LABELS_PATH, NOISE_DATASET_PATH, COMMANDS_DATASET_PATH


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default=os.path.join('.', 'data'),
                        help='The data directory (default: "./data").')
    parser.add_argument('--trigger-word', type=str, default='marvin',
                        help='The trigger word (default: "marvin").')
    parser.add_argument('--training-example-duration', type=int, default=12,
                        help='The duration of training examples in seconds (default: 12).')
    parser.add_argument('--max-num-pos', type=int, default=4,
                        help='The maximum number of positives in one example (default: 4).')
    parser.add_argument('--max-num-neg', type=int, default=5,
                        help='The maximum number of negatives in one example (default: 5).')
    parser.add_argument('--num-examples', type=int, default=20000,
                        help='Number of examples to generate (default: 20000).')

    return parser.parse_args()


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


def main():
    args = parse_args()

    command_dataset = CommandDataset(args)
    noise_dataset = NoiseDataset(args)
    training_examples_path = os.path.join(args.data_dir, TRAINING_EXAMPLES_PATH)
    os.makedirs(training_examples_path, exist_ok=True)
    training_labels_path = os.path.join(args.data_dir, TRAINING_LABELS_PATH)
    os.makedirs(training_labels_path, exist_ok=True)

    print('Generating dataset')
    for index in range(args.num_examples):
        positives_count = np.random.randint(low=0, high=args.max_num_pos)
        positives = [command_dataset.random_positive() for i in range(positives_count)]

        negatives_count = np.random.randint(low=0, high=args.max_num_neg)
        negatives = [command_dataset.random_negative() for j in range(negatives_count)]

        noise = noise_dataset.random()

        example, y = construct_example(noise, positives, negatives)
        torchaudio.save(os.path.join(training_examples_path, f'{index}.wav'), example.unsqueeze(0), 16000)
        torch.save(y, os.path.join(training_labels_path, f'{index}.pt'))

        print(f'\r{round(index / args.num_examples * 100, 2)} %', end='')


if __name__ == '__main__':
    main()
