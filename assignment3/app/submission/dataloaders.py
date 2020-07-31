from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_images = datasets.ImageFolder(
    './data/processed/train',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)

test_images = datasets.ImageFolder(
    './data/processed/test/',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)

train_dataloader = DataLoader(train_images, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_images, batch_size=10)