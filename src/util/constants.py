import os

RESULT_PATH = 'results'
CHECKPOINT_PATH = 'checkpoints'  # when editing, don't forget to add the path to .gitignore file

# dataset path constants
DATASETS_ROOT = 'data'  # when editing, don't forget to add the path to .gitignore file
NOISE_DATASET_PATH = os.path.join(DATASETS_ROOT, 'noise_dataset')
COMMAND_DATASET_PATH = os.path.join(DATASETS_ROOT, 'commands_dataset')
TRAINING_EXAMPLES_PATH = os.path.join(DATASETS_ROOT, 'training', 'examples')
TRAINING_LABELS_PATH = os.path.join(DATASETS_ROOT, 'training', 'labels')

# dataset generation constants
TRIGGER_WORD = 'marvin'
EXAMPLE_DURATION = 12
EXAMPLE_SAMPLING_RATE = 16000
MAX_POSITIVES_COUNT_IN_EXAMPLE = 4
MAX_NEGATIVES_COUNT_IN_EXAMPLE = 5
EXAMPLES_COUNT = 20000

# data transformation constants
FFT_SIZE = 400
FFT_WIN_LENGTH = 200
FFT_HOP_LENGTH = 100

# model constants
CONV_IN_CHANNELS = FFT_SIZE // 2 + 1
CONV_OUT_CHANNELS = 196
CONV_KERNEL_SIZE = 15
CONV_STRIDE = 4
FIRST_GRU_HIDDEN_SIZE = 128
SECOND_GRU_HIDDEN_SIZE = 128

# training constants
BATCH_SIZE = 256
EPOCHS_NUMBER = 40
LEARNING_RATE = 1e-3
