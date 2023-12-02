import os

# when editing, don't forget to add the path to .gitignore file
DATASETS_ROOT = 'data'
NOISE_DATASET_PATH = os.path.join(DATASETS_ROOT, 'noise_dataset')
COMMAND_DATASET_PATH = os.path.join(DATASETS_ROOT, 'commands_dataset')
BACKGROUND_NOISE_DURATION = 12
SAMPLING_RATE = 16000
TRIGGER_WORD = 'marvin'
TRAINING_DATASET_PATH = os.path.join(DATASETS_ROOT, 'training')
TRAINING_EXAMPLES_COUNT = 20000
MAX_POSITIVES_COUNT_IN_EXAMPLE = 4
MAX_NEGATIVES_COUNT_IN_EXAMPLE = 5
