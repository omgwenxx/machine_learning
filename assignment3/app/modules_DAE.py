import torch
from torch import nn, optim
import numpy as np
from dataloaders_DAE import train_dataloader_DAE1, train_dataloader_DAE2, train_dataloader_DAES, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images, AddNoise, Autoencoder
import time

def buildDAESoftmaxModel(model, lRate, epochs, plot=False, verbose=False):

    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print("Starting at " + timeStr + " to build " + model.name + " model...")
    
    autoencoder = Autoencoder(2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lRate)

    train_losses, validate_losses, accuracy_data = [], [], []

    valid_loss_min = np.Inf
    for _ in range(epochs):
        _ += 1
        if (_%100==0): print("epoch: " + str(_))
        running_loss = 0
        for images, labels in train_dataloader_DAES:
            
            images = autoencoder(images)
            
            output = model(images)

            loss = criterion(output, labels)
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
                    log_ps = model(images)
                    validate_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()
            train_loss = running_loss / len(train_dataloader_DAES)
            valid_loss = validate_loss / len(test_dataloader.dataset)

            train_losses.append(train_loss)
            validate_losses.append(valid_loss)
            accuracy_data.append(accuracy / len(test_dataloader))

            if (verbose):
                print("Epoch: {}/{}.. ".format(_, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader_DAES)),
                  "Validate Loss: {:.3f}.. ".format(validate_loss / len(test_dataloader)),
                  "Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))

            if (valid_loss <= valid_loss_min):
                if (verbose):
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        valid_loss_min, valid_loss))
                torch.save(model.state_dict(), './models/DAESoftMax_model.pt')
                valid_loss_min = valid_loss

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print("Finished at " + timeStr + ", duration in sec: " + str(int(dur)))

    if (plot):
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle("Model: " + model.name)
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
    
    train_dataloader = train_dataloader_DAE1 if "1000" in model.name else train_dataloader_DAE2
    train_dataloader = train_dataloader_DAE1
    
    noiser = AddNoise(0.2) if "1000" in model.name else AddNoise(0.3)
    autoencoder = Autoencoder(1)
    
    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print("Starting at " + timeStr + " to build " + model.name + " model...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lRate)

    train_losses, validate_losses, accuracy_data = [], [], []

    valid_loss_min = np.Inf
    for _ in range(epochs):
        _ += 1
        if (_%100==0): print("epoch: " + str(_))
        running_loss = 0
        for images, labels in train_dataloader:
            
            if ("300" in model.name):
                images = autoencoder(images)
            noised_images = noiser(images)
                
            # train_img = torch.Tensor(images)[0].view(112,92).detach().numpy()
            # plt.imshow(train_img, cmap='gray')
            # plt.show()
             
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
                    log_ps = model(images)
                    images_reshaped = images.view(images.shape[0], -1)
                    validate_loss += criterion(log_ps, images_reshaped)

            model.train()
            train_loss = running_loss / len(train_dataloader)
            valid_loss = validate_loss / len(test_dataloader.dataset)

            train_losses.append(train_loss)
            validate_losses.append(valid_loss)

            if (verbose):
                print("Epoch: {}/{}.. ".format(_, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
                  "Validate Loss: {:.3f}.. ".format(validate_loss / len(test_dataloader)))

            if (valid_loss <= valid_loss_min):
                if (verbose):
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        valid_loss_min, valid_loss))
                torch.save(model.state_dict(), './models/'+model.name+"_model.pt")
                valid_loss_min = valid_loss

    endTime = time.time()
    dur = endTime - startTime
    timeStr = time.strftime("%H:%M:%S", time.localtime(endTime))
    print("Finished at " + timeStr + ", duration in sec: " + str(int(dur)))

    if (plot):
        train_img = torch.Tensor(images)[2].view(112,92)
        rec_img = log_ps[2].view(112,92).detach().numpy()
                      
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle("Model: " + model.name)
                      
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

def evaluateModel_DAE(model, img_nr):
    
    model.load_state_dict(torch.load('./models/' + model.name + '_model.pt'))

    images = torch.Tensor(get_orig())
    output = model(images)
    
    train_img = torch.Tensor(images)[img_nr].view(112,92)
    rec_img = output[img_nr].view(112,92).detach().numpy()
                  
    fig = plt.figure(figsize=(10, 3))
    fig.suptitle("Model: " + model.name)
                  
    ax2 = fig.add_subplot(1,2,1)
    ax2.imshow(train_img, cmap='gray')
                  
    ax3 = fig.add_subplot(1,2,2)
    ax3.imshow(rec_img, cmap='gray')
                  
    plt.show()