print("Hello!")

from utils import get_classes
import os


classes = get_classes()
print(classes)
classes = os.listdir('./data/processed/train')
c_to_i = lambda x: classes.index(x)
i_to_c = lambda x: classes[x]
print(classes)