import os

from src.datasets.trigger_word_detection_dataset import TriggerWordDetectionDataset
from src.util.constants import TRAINING_EXAMPLES_PATH, TRAINING_LABELS_PATH, RESULT_PATH

import matplotlib.pyplot as plt
import torch
import numpy as np

project_root = os.path.abspath(os.path.join('..', '..'))

model_path = os.path.join(project_root, RESULT_PATH, 'model.pt')
model = torch.load(model_path)
model.eval()

dataset = TriggerWordDetectionDataset(os.path.join(project_root, TRAINING_EXAMPLES_PATH),
                                      os.path.join(project_root, TRAINING_LABELS_PATH))
random_index = np.random.randint(low=0, high=len(dataset))
audio, label = dataset[random_index]
audio = audio[None, :, :, :]  # emulate batch
with torch.no_grad():
    prediction = model(audio)[0]

figure, axis = plt.subplots(1, 2)

axis[0].plot(label.numpy())
axis[0].set_title('Label')

axis[1].plot(prediction.numpy())
axis[1].set_title('Prediction')

plt.show()
