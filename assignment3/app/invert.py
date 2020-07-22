import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from settings import DEBUG, RESIZE
from networks import CNNetwork, SMNetwork, MLPNetwork


pred = ''
def invert(model, img, lr, c, best_loss, best_x, i):
    img = torch.Tensor(img).view(1, -1)
    if not img.requires_grad:
        img.requires_grad = True
        
    optimizer.zero_grad()
    pred = model(img)
    loss = criterion(pred, torch.LongTensor([c]))
    loss.backward()    
    img = torch.clamp(img - lr * img.grad, 0, 255)

    if loss.detach().numpy() < best_loss and i > 10:
        best_loss = loss.detach().numpy()
        best_x = img.detach().numpy()
        
    filt = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])
    np_a = np.array([np.clip(x + np.random.normal(2, 2),0,255) for x in img.detach().numpy()])
    i = convolve(np_a.reshape(112, 92), filt)
    
    return best_loss, best_x, np_a.reshape(1, -1)

netName = "SM"
network = SMNetwork()
network.load_state_dict(torch.load(netName+'Network.pt'))  
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01)

classes = os.listdir('./data/processed/test')
c_to_i = lambda x: classes.index(x)
i_to_c = lambda x: classes[x]

test = 0
test_x = ([0]) * 280

for c in classes:
    for faces in os.listdir('./data/processed/test/'+c):
        img = np.array(Image.open('./data/processed/test/'+c+'/'+faces).convert('L'))
        test_x[test] = img.flatten()
        test += 1

for c in classes:
    best_x,best_loss='',float('inf')
    img = np.zeros_like(test_x[0])
    for i in range(100):
        best_loss,best_x,img = invert(network, img, 0.01, c_to_i(c), best_loss, best_x, i)
        if (i%10==0): print("i: " + str(i) + ", best_loss. " + str(best_loss))
    if c=='s34':
        rec = best_x.reshape(112, 92)
        orig = test_x[c_to_i(c)].reshape(112, 92).astype('float32')
        ssmv = ssm(rec,orig)
        msev = mse(rec,orig)
        nrmsev = nrmse(rec,orig)
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle("SSM: {:.4f}, MSE: {:.3f}, NRMSE: {:.3f}".format(ssmv,msev,nrmsev))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(rec, cmap='gray')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(orig, cmap='gray')
        # plt.savefig(f'./data/results/class {c}.png')
        plt.show()
        break