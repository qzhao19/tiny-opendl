# Lightweight-Deep-Learning-Framework (ing...)
A lightweight Deep learning framework using Python 

# Overview 

A lightweight deep learning framework using Python language, which enables create computation graph for gradient computation by automatic differentiation algorithm. It implements the sequence model, computation graph model, and numerous neural network building blocks such as layers, activation functions, optimizers.

# Requirement

- Python 3.6
- NumPy 1.16.5

# Classes

- `Model`  
  - computational graph 
  - Sequential model
- `Layers`
  - Add
  - Activation
  - Conv2D
  - Dense
  - Dropout
  - Flatten
  - MaxPooling
  - Reshape
- `Losses`
  - MSE(Mean Squar Error)
  - SoftmaxCrossEntropy
- `Optimizers`
  - SGD
  - Momentum

# Example

### Sequential model example

`
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np

data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')

model = Sequential(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
model.add(Dense(512, X.shape[-1]))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))

errs, accs = model.fit(X, Y, 10)
plots(errs, accs)
`
