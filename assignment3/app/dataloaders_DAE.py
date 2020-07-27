from torch.utils.data import DataLoader
from torch import from_numpy
from torchvision import datasets, transforms
import numpy as np
from utils import AddNoise, Autoencoder
    
train_images_DAE1 = datasets.ImageFolder(
    './data/processed/train',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
)

train_images_DAE2 = datasets.ImageFolder(
    './data/processed/train',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
)


train_images_DAES = datasets.ImageFolder(
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

#train_dataloader_DAE1 = DataLoader(train_images_DAE1, batch_size=len(train_images_DAE1), shuffle=True)
#train_dataloader_DAE2 = DataLoader(train_images_DAE2, batch_size=len(train_images_DAE2), shuffle=True)
#train_dataloader_DAES = DataLoader(train_images_DAES, batch_size=len(train_images_DAES), shuffle=True)
#test_dataloader = DataLoader(test_images, batch_size=len(test_images))

train_dataloader_DAE1 = DataLoader(train_images_DAE1, batch_size=20, shuffle=True)
train_dataloader_DAE2 = DataLoader(train_images_DAE2, batch_size=20, shuffle=True)
train_dataloader_DAES = DataLoader(train_images_DAES, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_images, batch_size=20)