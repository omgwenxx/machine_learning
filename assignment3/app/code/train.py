from networks import CNN, SoftMax, MLP
from modules import buildModel, reconstructionAttack

print("Building Models")
print("SoftMax")
buildModel(SoftMax(), 0.1, True, True)
    
print("MLP")
buildModel(MLP(), 0.1, True, True)

print("DAE")
buildDAELayer(DAELayer(10304, 1000), lRate=1e-4, epochs=5000, plot=True)
buildDAELayer(DAELayer(1000, 300), lRate=1e-4, epochs=5000, plot=True)
buildDAESoftmaxModel(DAESoftMax(), lRate=1e-2, epochs=1000, plot=True)

print("CNN")
buildModel(CNN(), 0.001, True, True)