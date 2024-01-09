import argparse
import os.path
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.datasets.trigger_word_detection_dataset import TriggerWordDetectionDataset
from src.model.trigger_word_detection_model import TriggerWordDetectionModel
from src.util.constants import *
from src.util.device import to_device

summary_writer = SummaryWriter()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default=os.path.join('.', 'data'),
                        help='The data directory (default: "./data").')
    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='The directory to save checkpoints to (default: "./checkpoints").')

    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs (default: 40).')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Base learning rate (default: 0.001).')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256).')
    parser.add_argument('--n-fft', type=int, default=400,
                        help='FFT size (default: 400).')
    parser.add_argument('--fft-window-length', type=int, default=200,
                        help='FFT window length (default: 200).')
    parser.add_argument('--fft-hop-length', type=int, default=100,
                        help='FFT hop length (default: 100).')
    parser.add_argument('--filters', type=int, default=196,
                        help='The number of convolutional layer filters (default: 196).')
    parser.add_argument('--kernel-size', type=int, default=15,
                        help='The kernel size of the convolutional layer (default: 15).')
    parser.add_argument('--stride', type=int, default=4,
                        help='The stride of the convolutional layer (default: 4).')
    parser.add_argument('--gru-1', type=int, default=128,
                        help='First GRU hidden units (default: 128).')
    parser.add_argument('--gru-2', type=int, default=128,
                        help='Second GRU hidden units (default: 128).')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize an example at the end of each testing phase.')

    return parser.parse_args()


def update_args(args):
    args.fft_freq_bins = args.n_fft // 2 + 1
    return args


def fit(epochs, lr, model, train_loader, test_loader, opt_fn, loss_fn, checkpoints_dir, should_visualize):
    optimizer = opt_fn(model.parameters(), lr)
    for epoch in range(1, epochs + 1):
        model.train(True)
        training_loss_sum = 0
        for batch in train_loader:
            batch = to_device(batch)
            x, y = batch
            prediction = model(x)
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item()
        print_loss('Training loss', training_loss_sum / len(train_loader), epoch)
        evaluate(model, test_loader, loss_fn, epoch, should_visualize)
        checkpoint(model, f'epoch-{epoch}.pt', checkpoints_dir)


def evaluate(model, test_loader, loss_fn, epoch, should_visualize):
    model.eval()
    test_loss_sum = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch)
            x, y = batch
            prediction = model(x)
            loss = loss_fn(prediction, y)
            test_loss_sum += loss.item()
    print_loss('Test loss', test_loss_sum / len(test_loader), epoch)
    if should_visualize:
        batch_size = prediction.size(0)
        rand_index = random.randint(0, batch_size - 1)
        visualize(prediction[rand_index], y[rand_index], epoch)


def visualize(prediction, label, epoch):
    figure, axis = plt.subplots(1, 2)

    axis[0].plot(label.cpu().numpy())
    axis[0].set_title(f'Label (Epoch: {epoch})')

    axis[1].plot(prediction.cpu().numpy())
    axis[1].set_title(f'Prediction (Epoch: {epoch})')

    plt.show()


def print_loss(tag, loss, epoch):
    summary_writer.add_scalar(tag, loss, epoch)
    print(f'Epoch: {epoch}, {tag}: {loss}')


def checkpoint(model, filename, checkpoints_dir):
    torch.save(model, os.path.join(checkpoints_dir, filename))
    torch.save(model, MODEL_FILENAME)


def main():
    args = update_args(parse_args())

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    model = TriggerWordDetectionModel(args)
    to_device(model)
    training_set, test_set = random_split(TriggerWordDetectionDataset(args), [0.8, 0.2])
    training_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    fit(args.epochs, args.learning_rate, model, training_loader, test_loader, torch.optim.Adam,
        torch.nn.functional.binary_cross_entropy, args.checkpoints_dir, args.visualize)


if __name__ == '__main__':
    main()
