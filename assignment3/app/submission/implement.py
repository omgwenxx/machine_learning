import torch
from networks import LogSoftMax, ConvNet
from modules_multi import buildModelMulti, invertModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)



buildModelMulti(LogSoftMax(), lRate=0.01, epochs=30, plot=True, save=True)
invertModel(LogSoftMax(), lrMod=0.001, lrInv=0.001, nStep=50, plot=True, save=True)

buildDAELayer(DAELayer(10304, 1000), lRate=1e-4, epochs=5000, plot=True, save=True)
buildDAELayer(DAELayer(1000, 300), lRate=1e-4, epochs=5000, plot=True, save=True)
buildDAESoftmaxModel(DAESoftMax(), lRate=1e-2, epochs=1000, plot=True, save=True)

buildModelMulti(ConvNet(), 0.001, 50, plot=True, save=True)
invertModel(ConvNet(), 0.01, 0.01, 50, plot=True, save=True)