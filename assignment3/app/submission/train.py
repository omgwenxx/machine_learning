from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

print("Building Models")
print("SoftMax")
buildModel(SoftMax(), 0.1, True, True)
    
print("MLP")
buildModel(MLP(), 0.1, True, True)
    
print("CNN")
buildModel(CNN(), 0.001, True, True)