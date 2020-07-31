import torch
import sys
import argparse
from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

# reconstructionAttack(model, alpha = 5000, beta = 100, gamme = 0.1, delta = 0.1, save = True, show = False)
# parameters can be adjusted

# SoftMax Model from paper
print("Softmax")
reconstructionAttack(SoftMax())

# MLP Model from paper
print("MLP")
reconstructionAttack(MLP(), beta=1000)

# CNN for comparison
print("CNN")
#reconstructionAttack(CNN())