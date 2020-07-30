import torch
from torch import nn, optim
import numpy as np
from dataloaders import train_dataloader, test_dataloader
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssm
from utils import classes, c_to_i, get_orig, show_images
import time

def buildModel(model, lRate, epochs, plot=False, verbose=False):

    startTime = time.time()
    timeStr = time.strftime("%H:%M:%S", time.localtime(startTime))
    print("Starting at " + timeStr + " to build " + model.name + " model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lRate)

    train_losses, validate_losses, accuracy_data = [], [], []

    valid_loss_min = np.Inf
    for _ in range(epochs):
        _ += 1
        if (_%100==0): print("epoch " + str(_) + " at " + time.strftime("%H:%M:%S", time.localtime(time.time())))
        running_loss = 0
        for images, labels in train_dataloader:
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
            train_loss = running_loss / len(train_dataloader)
            valid_loss = validate_loss / len(test_dataloader.dataset)

            train_losses.append(train_loss)
            validate_losses.append(valid_loss)
            accuracy_data.append(accuracy / len(test_dataloader))

            if verbose:
                print("Epoch: {}/{}.. ".format(_, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
                  "Validate Loss: {:.3f}.. ".format(validate_loss / len(test_dataloader)),
                  "Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))

            if valid_loss <= valid_loss_min:
                if verbose:
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        valid_loss_min, valid_loss))
                torch.save(model.state_dict(), './models/'+model.name+"_model.pt")
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


def invert_one(model, crit, optim, img, lr, c, best_loss, best_x, i):
    img = torch.Tensor(img)
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


def invert(model, lrMod, lrInv, nStep=20, plot=False, verbose=False,
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
            best_loss,best_x,img = invert_one(model, crit, optim, img, lrInv,
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
        plt.show()

        print("SSM: mean {:.2e}, std {:.2e}".format(np.mean(ssm_vs),np.std(ssm_vs)))
        print("NRMSE: mean {:.2e}, std {:.2e}".format(np.mean(nrmse_vs),np.std(nrmse_vs)))

        show_images(np.concatenate((test_x[0:5],rec_x[0:5]), axis=0),"Model: "+model.name)
    
    # return dur, rec_x, ssm_vs, nrmse_vs