from torch.utils.data import DataLoader
from torch import from_numpy
from torchvision import datasets, transforms
import numpy as np
from utils import AddNoise, Autoencoder
    
train_images = datasets.ImageFolder(
    './data/processed/train',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
)
test_images = datasets.ImageFolder(
    './data/processed/test',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)

train_dataloader = DataLoader(train_images, batch_size=len(train_images), shuffle=True)
test_dataloader = DataLoader(test_images, batch_size=len(test_images))

# train_dataloader = DataLoader(train_images, batch_size=20, shuffle=True)
# test_dataloader = DataLoader(test_images, batch_size=20)