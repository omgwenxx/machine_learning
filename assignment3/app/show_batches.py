from dataloaders import train_dataloader, test_dataloader
from settings import SHOW_BATCHES
from utils import show_batch

if SHOW_BATCHES['train']:
    nBat = 0
    for X, y in train_dataloader:
        nBat += 1
        show_batch(X, y, ('Train batch ' + str(nBat)))

if SHOW_BATCHES['test']:
    nBat = 0
    for X, y in test_dataloader:
        nBat += 1
        show_batch(X, y, ('Test batch ' + str(nBat)))
