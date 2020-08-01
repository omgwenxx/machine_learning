import torch
from networks import LogSoftMax, CNN, SoftMax, MLP, DAELayer, DAESoftMax
from modules import reconstructionAttack
from modules_DAE import  buildDAELayer, buildDAESoftmaxModel
import torch
import sys
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)
#buildDAELayer(DAELayer(10304, 1000), lRate=1e-4, epochs=5000, plot=True)
#buildDAELayer(DAELayer(1000, 300), lRate=1e-4, epochs=5000, plot=True)
#buildDAESoftmaxModel(DAESoftMax(), lRate=1e-2, epochs=1000, plot=True)

reconstructionAttack(DAESoftMax())

