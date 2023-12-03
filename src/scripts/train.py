import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.model.dataset import Dataset
from src.model.trigger_word_detection_model import TriggerWordDetectionModel
from src.util.constants import *
from src.util.device import to_device

project_root = os.path.abspath(os.path.join('..', '..'))
summary_writer = SummaryWriter()
EPOCHS_NUMBER = 40
LEARNING_RATE = 1e-3
os.makedirs(os.path.join(project_root, CHECKPOINT_PATH), exist_ok=True)
checkpoint_path = os.path.join(project_root, CHECKPOINT_PATH, 'model.pt')


def evaluate(model, val_loader, epoch):
    model.eval()
    output_sum = 0
    with torch.no_grad():
        for batch in val_loader:
            to_device(batch)
            output = model.calculate_loss(batch)
            output_sum += output.item()
    summary_writer.add_scalar('validation loss', output_sum / len(val_loader), epoch)


def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train(True)
        training_loss_sum = 0
        for batch in train_loader:
            to_device(batch)
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item()
        summary_writer.add_scalar('training loss', training_loss_sum / len(train_loader), epoch)
        evaluate(model, val_loader, epoch)
        torch.save(model, checkpoint_path)


if os.path.exists(checkpoint_path):
    model = torch.load(checkpoint_path)
else:
    model = TriggerWordDetectionModel()

to_device(model)
training_set, validation_set = random_split(
    Dataset(os.path.abspath(os.path.join(project_root, TRAINING_EXAMPLES_PATH)),
            os.path.abspath(os.path.join(project_root, TRAINING_LABELS_PATH))),
    [0.8, 0.2])
training_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=True)
fit(EPOCHS_NUMBER, LEARNING_RATE, model, training_loader, validation_loader, torch.optim.Adam)
