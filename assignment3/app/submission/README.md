# Face Recognition Model Inversion

This project build with different files. Separate python files are run to achieve different tasks.

1. prepare.py - *Use to separate raw dataset into desirable ratio.*
2. main.py - *Training model with processed data and saving model in .pt file and attacking models with default values*
2. train.py - *Training model with processed data*
2. reconstruct.py - *TAttacking models, parameters can be set by user*
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


## Main

Running training and inversion of models with default values.

```bash 
python main.py
```

## Train

Running training of models. Learning Rate can be set manually in the train.py file setting lRate within the function call of buildModel.

```bash 
python train.py
```

## Reconstruct

Running inversion of models. Following parameters can be set and are otherwise set to the default values specified in the given order:

* -m model: model attacked, valid values are 'Softmax', 'MLP', 'DAE', 'CNN', 'all' (default 'all'), so either one model can be attacked or all, consider that this parameter is case sensitive
* -a alpha: number of epochs the SDG runs through (default 5000)
* -b beta: number of iterations that the algorithm waits for improvement (default 100)
* -g gamma: constraint value for the cost, the algorithm stops once the cost value is below (default 0.01), should be a value in a range of [0,1]
* -d delta: learing rate of the SDG (0.1)

All of the parameters need to be set! The default values are mentioned for reference.

```bash 
python reconstruct.py -m all -a 5000 -b 100 -g 0.01 -d 0.1
```

code framework adapted from https://github.com/roshanshrestha01/face-recgonition-cnn
many thanx!

 
