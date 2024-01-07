import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.datasets.trigger_word_detection_dataset import TriggerWordDetectionDataset
from src.model.trigger_word_detection_model import TriggerWordDetectionModel
from src.util.constants import *
from src.util.device import to_device

summary_writer = SummaryWriter()


def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(1, epochs + 1):
        model.train(True)
        training_loss_sum = 0
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item()
        print_loss('Training loss', training_loss_sum / len(train_loader), epoch)
        evaluate(model, val_loader, epoch)
        checkpoint(model, f'epoch-{epoch}.pt')


def evaluate(model, val_loader, epoch):
    model.eval()
    output_sum = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = to_device(batch)
            output = model.calculate_loss(batch)
            output_sum += output.item()
    print_loss('Validation loss', output_sum / len(val_loader), epoch)


def print_loss(tag, loss, epoch):
    summary_writer.add_scalar(tag, loss, epoch)
    print(f'Epoch: {epoch}, {tag}: {loss}')


def checkpoint(model, filename):
    torch.save(model, os.path.join(CHECKPOINT_PATH, filename))
    torch.save(model, os.path.join(RESULT_PATH, 'model.pt'))


def main():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)

    model = TriggerWordDetectionModel()
    to_device(model)
    training_set, validation_set = random_split(
        TriggerWordDetectionDataset(TRAINING_EXAMPLES_PATH, TRAINING_LABELS_PATH),
        [0.8, 0.2])
    training_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=True)
    fit(EPOCHS_NUMBER, LEARNING_RATE, model, training_loader, validation_loader, torch.optim.Adam)


if __name__ == '__main__':
    main()
