# Face Recognition Model Inversion

1. prepare.py - *Use to separate raw dataset into desirable ratio.*
2. inversion_softmax.ipynb - *Training and inverting softmax model with processed data*
3. inversion_MLP.ipynb - *Training and inverting MLP model with processed data*
4. inversion_dae.ipynb - *Training and inverting DAE model with processed data*
5. inversion_CNN.ipynb - *Training and inverting CNN model with processed data*


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

 
