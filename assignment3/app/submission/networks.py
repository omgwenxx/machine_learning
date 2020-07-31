import torch.nn.functional as F
from torch import nn, sigmoid, tanh
import torch

class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.name = "SoftMax"
        self.linear1 = nn.Linear(10304, 40)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return x
    
class LogSoftMax(nn.Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()
        self.name = "LogSoftMax"
        self.linear1 = nn.Linear(10304, 40)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.linear1(x), dim=1)
        return x

    
class DAESoftMax(nn.Module):
    def __init__(self):
        super(DAESoftMax, self).__init__()
        self.name = "DAESoftMax"
        self.linear = nn.Linear(300, 40)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.linear(x), dim=1)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.linear1 = nn.Linear(10304, 3000)
        torch.manual_seed(666)
        torch.nn.init.normal_(self.linear1.weight)
        self.linear2 = nn.Linear(3000, 40)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


class DAELayer(nn.Module):
    def __init__(self, vis, hid):
        super(DAELayer, self).__init__()
        self.name = f"DAE_{hid}"
        self.encode = nn.Linear(vis, hid)
        self.decode = nn.Linear(hid, vis)
        
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = tanh(self.encode(x))
        x = self.decode(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(10304, 512)
        self.linear2 = nn.Linear(512, 40)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 112, 92)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
