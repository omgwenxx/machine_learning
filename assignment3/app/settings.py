import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

RESIZE = (112, 92)

SHUFFLE_BATCH = True
SHOW_BATCHES = {
    'train': False,
    'test': True
}

DEBUG = False
DEBUG_EPOCHS_VIEW_IMAGE = [10, 65, 97]


