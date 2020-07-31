import torch
from networks import LogSoftMax, CNN, SoftMax, MLP, DAELayer, DAESoftMax
from modules_multi import buildModelMulti,invertModel, buildModel, reconstructionAttack
from modules_DAE import  buildDAELayer, buildDAESoftmaxModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

# SoftMax Model from paper
#softmax = buildModel(SoftMax(), 0.1, True, True)
#reconstructionAttack(softmax)

# MLP Model from paper
#mlp = buildModel(MLP(), 0.1, True, True)
#reconstructionAttack(mlp)

# CNN for comparison
cnn = buildModel(CNN(), 0.1, True, True)
reconstructionAttack(cnn)

buildModelMulti(LogSoftMax(), lRate=0.01, epochs=30, plot=True, save=True)
invertModel(LogSoftMax(), lrMod=0.001, lrInv=0.001, nStep=50, plot=True, save=True)

buildDAELayer(DAELayer(10304, 1000), lRate=1e-4, epochs=5000, plot=True)
buildDAELayer(DAELayer(1000, 300), lRate=1e-4, epochs=5000, plot=True)
buildDAESoftmaxModel(DAESoftMax(), lRate=1e-2, epochs=1000, plot=True)

buildModelMulti(CNN(), 0.001, 50, plot=True, save=True)
invertModel(CNN(), 0.01, 0.01, 50, plot=True, save=True)
