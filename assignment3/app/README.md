# Face Recognition

This project build with different files. Separate python files are run to achieve different tasks.

1. settings.py -  *Hold all the settings for entire project.* 
2. prepare.py - *Use to separate raw dataset into desirable ratio.*
3. implement.py - *Training model with processed data and saving model in .pt file.*
4. dataloaders.py - *Contain pytorch data loading from process images.*
5. network.py - *CNN network build using pytorch.*
6. show_batches.py - *Displays batch images from dataloaders*


## Installation

Creation of a virtual environment

```p
virtualenv venv
source venv/bin/activate
```

Installation of required packages.

```p
pip install -r requirements.txt
```

## Data Split

Split data into test data and traing data set. Stores images in processed data with pass parameters.

```bash
python prepare.py 6 4
```

Above commands sets 6 images for training in train directory and 4 images for testing in test directory.


## Implement

Running train and validation of model.

```bash 
python implement.py
```


Trains model and output orl_databse_faces.pt when validation loss is decreased. Also gives confusion-matrix.xls.





 
