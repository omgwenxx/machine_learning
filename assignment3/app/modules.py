import torch
from torch import nn, optim
import numpy as np
from dataloaders import train_dataloader, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images
import time
import os
import torchvision.transforms as transforms
from PIL import Image

def buildModel(model, lRate, iCount=100, plot=False, verbose=False):

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
    while iteration_count < iCount:
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
        
    # return dur

def test(model):
        TEST_DIR = './data/processed/test/'
        TRAIN_DIR = './data/processed/train/'
        classes = os.listdir(TRAIN_DIR)
        c_to_i = lambda x: classes.index(x)
        i_to_c = lambda x: classes[x]

        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

        def show_img(im):
            plt.imshow(im.reshape(112, 92) / 2 + .5, cmap='gray')
            plt.show()
            
        def one_hot(x):
            vec = [0] *len(classes)
            vec[x] = 1
            return vec
            
        train =test= 0
        train_x, train_y = ([0]) * 280, ([0]) * 280
        test_x, test_y = ([0]) * 120, ([0]) * 120
        asd = []

        for c in os.listdir(TRAIN_DIR):
            for faces in os.listdir(TRAIN_DIR+c):
                img = np.array(Image.open(TRAIN_DIR+c+'/'+faces).convert('L'))
                train_x[train] = (img).flatten()
                train_y[train] = (c_to_i(c))
                train += 1

        for c in os.listdir(TEST_DIR):
            for faces in os.listdir(TEST_DIR+c):
                img = np.array(Image.open(TEST_DIR+c+'/'+faces).convert('L'))
                test_x[test] = img.flatten()
                test_y[test] = (c_to_i(c))
                test += 1

        test_x = np.stack([x.flatten()/255 for x in test_x])
        test_y = np.array(test_y,  dtype=np.int64)
        
        
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(torch.Tensor(test_x))
            _, predicted = torch.max(outputs.data, 1)
            total += torch.LongTensor(test_y).size(0)
            correct += (predicted == torch.LongTensor(test_y)).sum().item()

        print('Accuracy of the model ',model.name,' on the',total,'test images: %d %%' % (
            100 * correct / total))
        


def invertClass(model, crit, optim, img, lr, c, best_loss, best_x, i):
    img = torch.Tensor(img) #.view(1, -1)
    if not img.requires_grad:
        img.requires_grad = True

    optim.zero_grad()
    pred = model(img)
    loss = crit(pred, torch.LongTensor([c]))
    loss.backward()
    img = torch.clamp(img - lr * img.grad, 0, 255)

    if loss.detach().numpy() < best_loss and i > 4:
        best_loss = loss.detach().numpy()
        best_x = img.detach().numpy()

    np_a = np.array([np.clip(x + np.random.normal(2, 2),0,255) for x in img.detach().numpy()])

    return best_loss, best_x, np_a #.reshape(1, -1)


def invertModel(model, lrMod, lrInv, nStep=20, plot=False, verbose=False,
               show=False, save=False):

    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print("Starting at " + timeStr + " to invert " + model.name + "...")

    
    model.load_state_dict(torch.load('./models/' + model.name + '_model.pt'))
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lrMod)

    ssm_vs, nrmse_vs = [], []
    test_x = get_orig()
    rec_x = np.zeros((40,112,92), dtype='float32')
    for c in classes:
        best_loss = float('inf')
        best_x = img = np.zeros((1,112,92), dtype='float32')
        for i in range(nStep):
            best_loss,best_x,img = invertClass(model, crit, optim, img, lrInv,
                                              c_to_i(c), best_loss, best_x, i)
            if (verbose and i%5==0):
                print("i: " + str(i) + ", best_loss. " + str(best_loss))

        orig = test_x[c_to_i(c)]
        rec = best_x.reshape(112, 92)
        rec_x[c_to_i(c)] = rec
        ssm_v = ssm(rec,orig)
        nrmse_v = nrmse(rec,orig)
        ssm_vs.append(ssm_v)
        nrmse_vs.append(nrmse_v)

        if (show or save):
            fig = plt.figure(figsize=(10, 4))
            fig.suptitle("SSM: {:.1e}, NRMSE: {:.1f}".format(ssm_v,nrmse_v))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(rec, cmap='gray')
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(orig, cmap='gray')
        if show:
            plt.show()
        if save:
            plt.savefig(f'./data/results/class_{c}.png')
        # if (c=='s12'): break

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print("Finished at " + timeStr + ", duration in sec: " + str(int(dur)))
    
    if plot:
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle("Model: " + model.name)
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(ssm_vs)
        ax1.set_ylabel('Structural Similarity')
        ax1.set_xlabel('class index')
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(nrmse_vs)
        ax2.set_ylabel('Normalized Root MSE')
        ax2.set_xlabel('class index')
        #plt.savefig('./images/'+ model.name +'_mi_results.png', bbox_inches='tight', pad_inches = 0)
        plt.show()

        print("SSM: mean {:.2e}, std {:.2e}".format(np.mean(ssm_vs),np.std(ssm_vs)))
        print("NRMSE: mean {:.2e}, std {:.2e}".format(np.mean(nrmse_vs),np.std(nrmse_vs)))

        show_images(np.concatenate((test_x[0:5],rec_x[0:5]), axis=0),"Model: "+model.name)
    
    # return dur, rec_x, ssm_vs, nrmse_vs