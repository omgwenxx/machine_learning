import torch
from networks import SoftMax, MLP, ConvNet
from modules import buildModel, invert

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

buildModel(SoftMax(), 0.01, 20)
buildModel(MLP(), 0.0001, 100)
buildModel(ConvNet(), 0.001, 20)

invert(SoftMax(), 0.001, 0.001, 50)
invert(MLP(), 0.0001, 0.01, 20)
invert(ConvNet(), 0.01, 0.01, 20)