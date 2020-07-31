import torch
from torch import nn, optim
import numpy as np
from dataloaders import train_dataloader, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images, AddNoise, Autoencoder
import time
import torch.nn.functional as F
import collections
from scipy.ndimage import gaussian_filter, convolve
from skimage.metrics import mean_squared_error as mse
import os
import torchvision.transforms as transforms
from skimage.restoration import denoise_nl_means
from skimage.filters import unsharp_mask
from IPython.display import clear_output
from PIL import Image

def buildDAESoftmaxModel(model, lRate, epochs, plot=False, verbose=False):

    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print(f'Starting at {timeStr} to build {model.name} model...')
    
    encoder = Autoencoder(0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lRate)

    train_losses, validate_losses, accuracy_data = [], [], []

    valid_loss_min = np.Inf
    for _ in range(epochs):
        _ += 1
        if (_%100==0): print(f'epoch: {_}')
        running_loss = 0
        for images, labels in train_dataloader:
            images = encoder.encode(images)
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            validate_loss = 0
            accuracy = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for images, labels in test_dataloader:
                
                    images = encoder.encode(images)
                    
                    log_ps = model(images)
                    validate_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()
            train_loss = running_loss / len(train_dataloader)
            valid_loss = validate_loss / len(test_dataloader.dataset)

            train_losses.append(train_loss)
            validate_losses.append(valid_loss)
            accuracy_data.append(accuracy / len(test_dataloader))

            if (verbose):
                print(f'Epoch: {_}/{epochs}.. ',
                  f'Training Loss: {running_loss / len(train_dataloader):.3f}.. ',
                  f'Validate Loss: {validate_loss / len(test_dataloader):.3f}.. ',
                  f'Accuracy: {accuracy / len(test_dataloader):.3f}')

            if (valid_loss <= valid_loss_min):
                if (verbose):
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        valid_loss_min, valid_loss))
                torch.save(model.state_dict(), f'./models/{model.name}_model.pt')
                valid_loss_min = valid_loss

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print(f'Finished at {timeStr}, duration in sec: {int(dur)}')

    if (plot):
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle(f'Model: Stacked denoising Autoencoder')
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
        plt.savefig('SDAE_model.png')
        plt.show()
        
def buildDAELayer(model, lRate, epochs, plot=False, verbose=False):
    
    # declare preprocessing steps
    noiser = AddNoise(0.3) if '300' in model.name else AddNoise(0.2)
    encoder = Autoencoder(1) if '300' in model.name else Autoencoder(-1)
    
    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print(f'Starting at {timeStr} to build {model.name} model...')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lRate)

    train_losses, validate_losses, accuracy_data = [], [], []

    valid_loss_min = np.Inf
    for _ in range(epochs):
        
        _ += 1
        if (_%100==0): print(f'epoch: {_}')
        running_loss = 0
        for images, labels in train_dataloader:
            
            if ('300' in model.name):
                images = encoder.encode(images)
            noised_images = noiser(images)
                
             
            output = model(noised_images)
            images_reshaped = images.view(images.shape[0], -1)
            loss = criterion(output, images_reshaped)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            validate_loss = 0
            accuracy = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for images, labels in test_dataloader:
                    
                    if ('300' in model.name):
                        images = encoder.encode(images)
                
                    log_ps = model(images)
                    images_reshaped = images.view(images.shape[0], -1)
                    validate_loss += criterion(log_ps, images_reshaped)

            model.train()
            train_loss = running_loss / len(train_dataloader)
            valid_loss = validate_loss / len(test_dataloader.dataset)

            train_losses.append(train_loss)
            validate_losses.append(valid_loss)

            if (verbose):
                print(f'Epoch: {_}/{epochs}.. ',
                  f'Training Loss: {running_loss / len(train_dataloader):.3f}.. ',
                  f'Validate Loss: {validate_loss / len(test_dataloader):.3f}.. ')

            if (valid_loss <= valid_loss_min):
                if (verbose):
                    print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
                torch.save(model.state_dict(), f'./models/{model.name}_model.pt')
                valid_loss_min = valid_loss

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print(f'Finished at {timeStr}, duration in sec: {int(dur)}')

    if (plot):
        if ('300' in model.name):
                images = encoder.decode(images)
                log_ps = encoder.decode(log_ps)
                
        train_img = images[2].view(112,92).detach().numpy()
        rec_img = log_ps[2].view(112,92).detach().numpy()
                      
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle(f'Model: {model.name}')
                      
        ax1 = fig.add_subplot(1,3,1)
        ax1.plot(train_losses, label='Training loss')
        ax1.plot(validate_losses, label='Validation loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('epochs')
        ax1.legend(frameon=False)
                      
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(train_img, cmap='gray')
                      
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(rec_img, cmap='gray')
                      
        plt.show()
        
def evaluate_DAEs(img_nr, save=False):
    
    encoder1 = Autoencoder(1)
    encoder2 = Autoencoder(0)
    
    orig_images = torch.Tensor(get_orig())
    
    encoded1 = encoder1.encode(orig_images)
    decoded1 = encoder1.decode(encoded1)
    
    encoded2 = encoder2.encode(orig_images)
    decoded2 = encoder2.decode(encoded2)
    
    train_img = torch.Tensor(orig_images)[img_nr].view(112,92)
    rec_img1 = decoded1[img_nr].view(112,92).detach().numpy()
    rec_img2 = decoded2[img_nr].view(112,92).detach().numpy()
                  
    fig = plt.figure(figsize=(10, 3))
                  
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(train_img, cmap='gray')
    ax1.set_title('original image')
         
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(rec_img1, cmap='gray')
    ax2.set_title('image after first DAE')
                  
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(rec_img2, cmap='gray')
    ax3.set_title('image after both DAEs')
    
    if (save):
        plt.savefig(f'DAE_performance_{img_nr}.png')
    plt.show()
    
def evaluateModel_DAE(model, img_nr):
    
    model.load_state_dict(torch.load(f'./models/{model.name}_model.pt'))
    encoder = Autoencoder(1)
    
    orig_images = torch.Tensor(get_orig())
    images = orig_images
    if ('300' in model.name):
        images = encoder.encode(images)
    
    output = model(images)
    
    if ('300' in model.name):
        output = encoder.decode(output)
    
    train_img = torch.Tensor(orig_images)[img_nr].view(112,92)
    rec_img = output[img_nr].view(112,92).detach().numpy()
                  
    fig = plt.figure(figsize=(10, 3))
    fig.suptitle("Model: " + model.name)
                  
    ax2 = fig.add_subplot(1,2,1)
    ax2.imshow(train_img, cmap='gray')
                  
    ax3 = fig.add_subplot(1,2,2)
    ax3.imshow(rec_img, cmap='gray')
                  
    plt.show()
    
def testDAE(model):
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
            encoder = Autoencoder(0)
            input_img = encoder.encode(torch.Tensor(test_x))
            outputs = model(input_img)
            _, predicted = torch.max(outputs.data, 1)
            total += torch.LongTensor(test_y).size(0)
            correct += (predicted == torch.LongTensor(test_y)).sum().item()

        print('Accuracy of the model ',model.name,' on the',total,'test images: %d %%' % (
            100 * correct / total))
    
def process(tensor):
    encoder = Autoencoder(0)
    img = encoder.decode(tensor)
    img = img.view(112,92)
    
    img2 = denoise_nl_means(img.detach().numpy(), patch_size=5, patch_distance=5, h=0.3)
    
    img3 = unsharp_mask(img2, radius=2, amount=1, preserve_range=True)  
    
    output = encoder.encode(torch.Tensor([img3]))
    return output
    img = torch.Tensor(img) 

    # ¿processing evtl. hier
        
    if not img.requires_grad:
        img.requires_grad = True
    optim.zero_grad()
        
    pred = model(img)
        
    loss = crit(pred, torch.LongTensor([c]))
    loss.backward()
    img = torch.clamp(img - lr * img.grad, 0, 255)

    
    if (processing):
        img = process(img)
        
    if loss.detach().numpy() < best_loss and i > 4:
        best_loss = loss.detach().numpy()
        best_x = img.detach().numpy()

    np_a = np.array([np.clip(x + np.random.normal(2, 2),0,255) for x in img.detach().numpy()])

    return best_loss, best_x, np_a #.reshape(1, -1)


    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print(f'Starting at {timeStr} to invert {model.name}...')

    encoder = Autoencoder(0)
    
    model.load_state_dict(torch.load(f'./models/{model.name}_model.pt'))
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lrMod)

    ssm_vs, nrmse_vs = [], []
    original_imgs = torch.Tensor(get_orig())
    test_x = encoder.encode(original_imgs)
    rec_x = np.zeros((40,300), dtype='float32')
    for i, c in enumerate(classes):
        best_loss = float('inf')
        best_x = img = np.zeros((1,300), dtype='float32')
        for epoch in range(nStep):
            
            # clear_output(wait=True)
            # print("Starting at " + timeStr + " to invert " + model.name + "...")
            # print(f'class {c} ({i+1}/{len(classes)})')
            # print(f'\tepoch {epoch}')
            
            best_loss,best_x,img = invertClass(model, crit, optim, img, lrInv,
                                              c_to_i(c), best_loss, best_x, epoch, processing)
            if (verbose and epoch%5==0):
                print(f'epoch: {epoch}, best_loss. {best_loss}')

        orig = test_x[c_to_i(c)].detach().numpy()
        rec = best_x.reshape(300)
        rec_x[c_to_i(c)] = rec
        ssm_v = ssm(rec,orig)
        nrmse_v = nrmse(rec,orig)

        ssm_vs.append(ssm_v)
        nrmse_vs.append(nrmse_v)
        if (show or save):
            encoder = Autoencoder(0)
            
            rec_show = encoder.decode(torch.Tensor([rec])).view(112,92).detach().numpy()
            orig_show = encoder.decode(torch.Tensor([orig])).view(112,92).detach().numpy()
            fig = plt.figure(figsize=(10, 4))
            fig.suptitle("SSM: {:.1e}, NRMSE: {:.1f}".format(ssm_v,nrmse_v))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(rec_show, cmap='gray')
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(orig_show, cmap='gray')
        if show:
            plt.show()
        if save:
            plt.savefig(f'./data/results/class_{c}.png')
        # if (c=='s12'): break

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print(f'Finished at {timeStr}, duration in sec: {int(dur)}')
    
    if plot:
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle(f'Model: {model.name}')
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(ssm_vs)
        ax1.set_ylabel('Structural Similarity')
        ax1.set_xlabel('class index')
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(nrmse_vs)
        ax2.set_ylabel('Normalized Root MSE')
        ax2.set_xlabel('class index')
        plt.show()

        print("SSM: mean {:.2e}, std {:.2e}".format(np.mean(ssm_vs),np.std(ssm_vs)))
        print("NRMSE: mean {:.2e}, std {:.2e}".format(np.mean(nrmse_vs),np.std(nrmse_vs)))
        # print(test_x.shape)
        # print(torch.Tensor(rec_x).shape)
        encoder = Autoencoder(0)
        test_imgs = original_imgs[0:5]
        rec_imgs = encoder.decode(torch.Tensor(rec_x[0:5])).view(5,112,92).detach().numpy()

        show_images(np.concatenate((test_imgs,rec_imgs), axis=0),f'Model: {model.name}')   

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

# returns a vector with all costs for all labels
def my_cost(pred):
    cost = torch.ones(pred.shape) - pred
    return cost 

def invert(model, img, criterion, optimizer, lr, c, best_cost, best_x, i, b, beta, gamma, processing=False):
    
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
    
    if (processing):
        img = process(img)
    
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
     
# values from paper with adjusted gamma and cost function     
def reconstructionAttack(model, alpha = 5000, beta = 100, gamma = 0.01, delta = 0.1, save = True, show = False):
    dae = False
    if (model.name == 'DAESoftMax'):
        dae = True
    
    # reload model
    model.load_state_dict(torch.load('models/'+model.name+'_model.pt'))
    
    # performance measures
    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    mse_all, nrmsev_all, ssmv_all, epochs = [],[],[],[]
    
    # SDG
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=delta)
    
    test_x = get_orig()
    rec_x = np.zeros((40,112,92), dtype='float32')
    process = False
    
    print('DAE Flag is',dae)
    # if DAE images have different size and process is true
    encoder = Autoencoder(0)
    if (dae):
        test_x = encoder.encode(torch.Tensor(test_x))
        rec_x = np.zeros((40,300), dtype='float32')
        
    for c in classes:
        print('\nReconstructing class',c)
        best_x,best_cost='',1
        if(dae):
            img = np.zeros_like(test_x[0].detach().numpy())
        else:
            img = np.zeros_like(test_x[0])
        ssmv, msev,nrmsev = 0,0,0
        rec,orig = '',''
        
        if (dae):
            process = True
            best_x = img = np.zeros((1,300), dtype='float32')
        else:
            np.zeros_like(test_x[0])
            
        b = beta
        
        for i in range(alpha):
            best_cost, best_x, b, img, stop = invert(model, img, criterion, optimizer, 
                                                  delta, c_to_i(c), best_cost, best_x, i, 
                                                  b, beta, gamma, processing = process)
            if stop:
                epochs.append(i)
                break
        
        if(dae):
            orig = test_x[c_to_i(c)].detach().numpy()
            rec = best_x.reshape(300)
            rec_x[c_to_i(c)] = rec
            
        else:
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
            if (dae):
                rec = encoder.decode(torch.Tensor([rec])).view(112,92).detach().numpy()
                orig = encoder.decode(torch.Tensor([orig])).view(112,92).detach().numpy()
            
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
    