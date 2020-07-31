import sys
import argparse
from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

argumentList = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs=1)
parser.add_argument("-a", "--alpha", nargs=1)
parser.add_argument("-b", "--beta", nargs=1)
parser.add_argument("-g", "--gamma", nargs=1)
parser.add_argument("-d", "--delta", nargs=1)

args = parser.parse_args()
model = 'all'
alpha = 5000
beta = 100
gamma = 0.01
delta = 0.1
if args.model:
    model = argumentList
if args.alpha:
    alpha = argumentList
if args.beta:
    beta = argumentList
if args.gamma:
    gamma = argumentList
if args.delta:
    delta = argumentList