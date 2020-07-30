# Face Recognition Model Inversion

This project build with different files. Separate python files are run to achieve different tasks.

1. prepare.py - *Use to separate raw dataset into desirable ratio.*
2. implement.py - *Training model with processed data and saving model in .pt file.*
3. network.py - *network definitions using pytorch.*


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
python prepare.py 7 3
```

Above commands sets 7 images for training in train directory and 3 images for testing in test directory.


## Implement

Running training and inversion of models.

```bash 
python implement.py
```


code framework adapted from https://github.com/roshanshrestha01/face-recgonition-cnn
many thanx!

 
