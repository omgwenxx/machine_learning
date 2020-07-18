import sys

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_images(path):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), 0)
                    X.append(im)
                    y.append(subdirname)
                except IOError:
                    print("I/O error")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]


classes = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22', 's23',
           's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37',
           's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']

def show_batch(images, labels, title):
    size = int(len(images) / 2)
    plt.suptitle(title)
    for idx, im in enumerate(images):
        plt.subplot(2, size, idx + 1)
        plt.gca().set_title(str(classes[labels[idx]]))
        plt.axis('off')
        plt.imshow(im[0], cmap='gray', interpolation='bicubic')
    plt.show()


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    fix, ax = plt.subplots()
    ax.barh(np.arange(40), ps)
    # ax.set_aspect(0.1)
    ax.set_yticks(np.arange(40))
    ax.set_yticklabels(classes, size='small')
    ax.set_title('Class Probability')
    plt.xlabel('Probability')
    plt.ylabel('Subjects value')
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
