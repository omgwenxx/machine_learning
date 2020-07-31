import torch
from torch import nn, optim
import numpy as np
from dataloaders import train_dataloader, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images
import time
import torch.nn.functional as F
from torch import nn
import collections
from scipy.ndimage import gaussian_filter, convolve
import time
from dataloaders import train_dataloader, test_dataloader
from torch import nn, optim
import torch
from scipy.ndimage import gaussian_filter, convolve
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images

def buildModel(model, lRate, plot=True, verbose=True):
    
    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print("Starting at " + timeStr + " to build " + model.name + " model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lRate)
    train_losses, validate_losses, accuracy_data = [], [], []
    valid_loss_min = np.Inf
    
    best_accuracy = 0
    iteration_count = 0
    total_iteration = 0
    
    # as in the paper, if there is no improvment after 100 iterations
    # stop training
    while iteration_count < 100:
        total_iteration += 1
        if (total_iteration%100==0): print("epoch: " + str(total_iteration))
        running_loss = 0
        for images, labels in train_dataloader:
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # epoch over
        iteration_count +=1
        
        validate_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in test_dataloader:
                samples = model(images)
                pred = samples.argmax(dim=1)
                accuracy = (pred == labels).sum().item() / len(labels)
                
            model.train()
            train_loss = running_loss / len(train_dataloader)
            train_losses.append(train_loss)
            accuracy_data.append(accuracy)
            
            
            if accuracy > best_accuracy:  # if we improve our accuracy, set the iteration count to 0
                if verbose:
                    print("Epoch: {}.. ".format(total_iteration))
                    print('Accuracy increased ({:.2f} --> {:.2f}). Saving model ...'.format(
                        best_accuracy, accuracy))
                torch.save(model.state_dict(), './models/'+model.name+"_model.pt")
                iteration_count = 0
                best_accuracy = accuracy  # update best accuracy

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print("Finished at " + timeStr + ", duration in sec: " + str(int(dur)))
    print("Total number of iterations ", total_iteration,", with accuracy of ", best_accuracy)
    
    if (plot):
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle("Model: " + model.name)
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(train_losses, label='Training loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('epochs')
        ax1.legend(frameon=False)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(accuracy_data, label='Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('epochs')
        ax2.legend(frameon=False)
        plt.savefig("images/"+ model.name+"_model.png", bbox_inches='tight')
        plt.show()
    
    return model

def invert(model, img, criterion, optimizer, lr, c, best_cost, best_x, i, b, beta, gamma):
    img = torch.Tensor(img).view(1, -1)
    if not img.requires_grad:
        img.requires_grad = True
        
    optimizer.zero_grad()
    pred = model(img)
    b -= 1

    
    # calculate cost
    cost = my_cost(F.softmax(pred))
    cost = cost.detach().numpy().flatten()[c]
    
    # calculate loss
    loss = criterion(pred, torch.LongTensor([c]))
    loss.backward()  
    
    img = torch.clamp(img - lr * img.grad, 0, 255)
    np_a = np.array([np.clip(x + np.random.normal(2, 2),0,255) for x in img.detach().numpy()])
    
    if cost < best_cost:
        print("Cost was updated, is now", cost,"was", best_cost)
        best_cost = cost
        best_x = img.detach().numpy()
        b = beta
    
    if cost >= best_cost and b <= 0:
        print("Cost doesn't improve after", beta, "iterations with a best value of", best_cost)
        best_cost = cost
        best_x = img.detach().numpy()
        return best_cost, best_x, b, np_a.reshape(1, -1), True
    
    if cost <= gamma:
        print("Cost is lower than gamma with a value of ", cost)
        best_cost = cost
        best_x = img.detach().numpy()
        return best_cost, best_x, b, np_a.reshape(1, -1), True
    
    return best_cost, best_x, b, np_a.reshape(1, -1), False

# returns a vector with all costs for all labels
def my_cost(pred):
    cost = torch.ones(pred.shape) - pred
    return cost 
     
# values from paper with adjusted gamma and cost function     
def reconstructionAttack(model, alpha = 5000, beta = 100, gamma = 0.01, delta = 0.1, save = True, show = False):
    
    # reload model
    model.load_state_dict(torch.load('models/'+model.name+'_model.pt'))
    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    mse_all, nrmsev_all, ssmv_all, epochs = [],[],[],[]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=delta)
    test_x = get_orig()
    rec_x = np.zeros((40,112,92), dtype='float32')

    for c in classes:
        print('\nReconstructing class',c)
        best_x,best_cost='',1
        img = np.zeros_like(test_x[0])
        b = beta
        for i in range(alpha):
            best_cost, best_x, b, img, stop = invert(model, img, criterion, optimizer, 
                                                  delta, c_to_i(c), best_cost, best_x, i, 
                                                  b, beta, gamma)
            if stop:
                epochs.append(i)
                break
        orig = test_x[c_to_i(c)]
        rec = best_x.reshape(112, 92)
        rec_x[c_to_i(c)] = rec
        ssmv = ssm(rec,orig)
        msev = mse(rec,orig)
        nrmsev = nrmse(rec,orig)
        mse_all.append(msev)
        nrmsev_all.append(nrmsev)
        ssmv_all.append(ssmv)
        
        if (show or save):
            fig = plt.figure(figsize=(10, 4))
            fig.suptitle("SSM: {:.1e}, NRMSE: {:.1f}".format(ssmv,nrmsev))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(rec, cmap='gray')
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(orig, cmap='gray')
            plt.savefig(f'./data/results/'+model.name+'_class_'+c+'.png')
            if show:
                plt.show()
            
    endTime = time.time()
    dur = endTime - startTime
    print("Duration in sec: " + str(int(dur)))
    
    # Calculating means performance values of all images
    print("\nAverage performance",model.name)
    print('MSE mean',np.mean(mse_all), 'with std of ', np.std(mse_all))
    print('NRMSE mean',np.mean(nrmsev_all), 'with std of +/-', np.std(nrmsev_all))
    print('SSM mean',np.mean(ssmv_all), 'with std of +/-', np.std(ssmv_all))
    print('Epochs mean',np.mean(epochs), 'with std of +/-', np.std(epochs))
    