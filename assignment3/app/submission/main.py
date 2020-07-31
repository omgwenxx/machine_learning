import torch
import sys
import argparse
from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

print("Building Models")
print("SoftMax")
buildModel(SoftMax(), 0.1, True, True)
    
print("MLP")
buildModel(MLP(), 0.1, True, True)

print("CNN")
buildModel(CNN(), 0.001, True, True)


print("\nModel(s) is reconstructed with alpha =", 5000 ,"beta =", 100 ,"gamma =", 0.01 ,"delta =", 0.1)
print("Attacking Models")
# SoftMax Model from paper
print("Softmax")
reconstructionAttack(SoftMax())

# MLP Model from paper
print("MLP")
reconstructionAttack(MLP())

# CNN for comparison
print("CNN")
reconstructionAttack(CNN())