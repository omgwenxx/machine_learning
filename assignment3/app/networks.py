import torch.nn.functional as F
from torch import nn

from settings import RESIZE

class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        
        self.linear1 = nn.Linear(10304, 3000)
        self.linear2 = nn.Linear(3000, 40)
        
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=1)
        return x


class SMNetwork(nn.Module):

    def __init__(self):
        super(SMNetwork, self).__init__()
        self.fc1 = nn.Linear(10304, 40)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.fc1(x), dim=1)
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
        self.fc1 = nn.Linear(10304, 2048)
        self.fc2 = nn.Linear(2048, 40)
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
