import os
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import numpy as np
from dataloaders_DAE import train_dataloader, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from skimage.restoration import denoise_nl_means
from skimage.filters import unsharp_mask
from utils import classes, c_to_i, get_orig, show_images, AddNoise, Autoencoder
import time
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
            images = encoder(images)
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
                
                    images = encoder(images)
                    
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
                  f'Training Loss: {running_loss / len(train_dataloader_DAES):.3f}.. ',
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
        fig.suptitle(f'Model: {model.name}')
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(train_losses, label='Training loss')
        ax1.plot(validate_losses, label='Validation loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('epochs')
        ax1.legend(frameon=False)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(accuracy_data, label='Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('epochs')
        ax2.legend(frameon=False)
        plt.show()
        
    # return dur

def buildDAELayer(model, lRate, epochs, plot=False, verbose=False):
    
    # declare preprocessing steps
    noiser = AddNoise(0.3) if '300' in model.name else AddNoise(0.2)
    encoder = Autoencoder(1)
    
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
                images = encoder(images)
            noised_images = noiser(images)
                
            train_img = torch.Tensor(noised_images)[0].view(112,92).detach().numpy()
            plt.imshow(train_img, cmap='gray')
            plt.show()
             
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
                        images = encoder(images)
                
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
        train_img = torch.Tensor(images)[2].view(112,92)
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
        
    # return dur

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
    
    # clear_output(wait=True)
    # fig = plt.figure(figsize=(10, 3))
    # ax1 = fig.add_subplot(1,3,1)
    # ax1.imshow(img.view(112,92).detach().numpy(), cmap='gray')
    # ax2 = fig.add_subplot(1,3,2)
    # ax2.imshow(img2.reshape(112,92), cmap='gray')
    # ax3 = fig.add_subplot(1,3,3)
    # ax3.imshow(img3.reshape(112,92), cmap='gray')
    # plt.show()    
    
    output = encoder.encode(torch.Tensor([img3]))
    return output
    
    
def invertClass(model, crit, optim, img, lr, c, best_loss, best_x, i, processing):
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


def invertDAE(model, lrMod, lrInv, nStep=20, plot=False, verbose=False,
               show=False, save=False, processing=False):

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
    
    # return dur, rec_x, ssm_vs, nrmse_vs