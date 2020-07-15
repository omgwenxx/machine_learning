import torch.nn.functional as F
from torch import nn

from settings import RESIZE


class NNetwork(nn.Module):

    def __init__(self):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 784)
        self.fc3 = nn.Linear(784, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 40)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x


class CNNetwork(nn.Module):

    def __init__(self):
        super(CNNetwork, self).__init__()
        # 128x128
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 64x64
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 32x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 40)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
