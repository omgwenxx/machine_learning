import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from networks import DAELayer

TEST_DIR = './data/processed/test/'
classes = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
           's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
           's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39',
           's4', 's40', 's5', 's6', 's7', 's8', 's9']
c_to_i = lambda x: classes.index(x)
i_to_c = lambda x: classes[x]

def get_orig():
    test_x = np.zeros((40,112,92), dtype='float32')
    i = 0
    for c in classes:
        faces = os.listdir(TEST_DIR + c)
        test_x[i] = np.array(Image.open(TEST_DIR + c + '/' +faces[0]).convert('L')).astype('float32')
        i += 1
    return test_x

def show_images(images, title='Figure'):
    size = int(len(images) / 2)
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(title)
    for idx, im in enumerate(images):
        plt.subplot(2, size, idx + 1)
        plt.axis("off")
        plt.imshow(im, cmap='gray')
    plt.show()

class AddNoise(object):
    def __init__(self, corruption_lvl):
        self.lvl = corruption_lvl
        
    def __call__(self, tensor):
        rand = np.random.random(tensor.shape)
        rand = rand >= self.lvl
        new = torch.from_numpy(np.zeros(tensor.shape, dtype=np.float32))
        new[[rand]] += tensor[[rand]]
        return new
    
class Autoencoder(object):
    def __init__(self, stage=2):
        self.stage = stage
        if (self.stage > 0):
            self.first_layer = DAELayer(10304, 1000)
            if (os.path.exists(f'./models/{self.first_layer.name}_model.pt')):
                self.first_layer.load_state_dict(torch.load(f'./models/{self.first_layer.name}_model.pt'))
            
            if (self.stage > 1):
                self.second_layer = DAELayer(1000, 300)
                if (os.path.exists(f'./models/{self.second_layer.name}_model.pt')):
                    self.second_layer.load_state_dict(torch.load(f'./models/{self.second_layer.name}_model.pt'))

    def encode(self, tensor):
        new = tensor
        if (self.stage > 0):
            new = self.first_layer.encode(tensor.view(tensor.shape[0], -1))
            if (self.stage > 1):
                new = self.second_layer.encode(new.view(new.shape[0], -1))
            
        return new
    
    def decode(self, tensor): 
        new = tensor
        if (self.stage > 0):    
            new = self.second_layer.decode(tensor.view(tensor.shape[0], -1))
            if (self.stage > 1):
                new = self.first_layer.decode(new.view(new.shape[0], -1))
            
        return new
    
    def __call__(self, tensor):
        return self.encode(tensor)
            