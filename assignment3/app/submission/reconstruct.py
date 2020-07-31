import torch
import sys
import argparse
from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs=1, default='all')
parser.add_argument("-a", "--alpha", nargs=1, default=5000)
parser.add_argument("-b", "--beta", nargs=1, default=100)
parser.add_argument("-g", "--gamma", nargs=1, default=0.01)
parser.add_argument("-d", "--delta", nargs=1, default=0.1)

args = parser.parse_args()

model = str(args.model[0])
alpha = int(args.alpha[0])
beta = int(args.beta[0])
gamma = float(args.gamma[0])
delta = float(args.delta[0])

print("\nModel(s) is reconstructed with alpha =", alpha ,"beta =", beta ,"gamma =", gamma ,"delta =", delta)
print("Models choosen",model)

if model == 'all':
    # SoftMax Model from paper
    print("Softmax")
    #reconstructionAttack(SoftMax(), alpha, beta, gamma, delta, True, False)

    # MLP Model from paper
    print("MLP")
    #reconstructionAttack(MLP(), alpha, beta, gamma, delta, True, False)
    
    # MLP Model from paper
    print("DAE")
    #reconstructionAttack(DAE(), alpha, beta, gamma, delta, True, False)

    # CNN for comparison
    print("CNN")
    #reconstructionAttack(CNN(), alpha, beta, gamma, delta, True, False)
else:
    if model == 'Softmax':
        # SoftMax Model from paper
        print("Softmax")
        #reconstructionAttack(SoftMax(), alpha, beta, gamma, delta, True, False)
    
    if model == 'MLP':
        # MLP Model from paper
        print("MLP")
        #reconstructionAttack(MLP(), alpha, beta, gamma, delta, True, False)
        
    if model == 'DAE':
        # MLP Model from paper
        print("DAE")
        #reconstructionAttack(DAE(), alpha, beta, gamma, delta, True, False)

    if model == 'CNN':
        # CNN for comparison
        print("CNN")
        #reconstructionAttack(CNN(), alpha, beta, gamma, delta, True, False)