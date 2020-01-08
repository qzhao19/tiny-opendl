from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Optimizer(object):
    """Optimizer base class

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    """
    def __init__(self):
        pass

    def set_layers(self, layers):
        self.layers = layers
    
    def update_layers(self):
        """Update params in sub-class method 
        """
        raise NotImplementedError
    
    def get_params(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Attributs:
        learning_rate: Float, the learning rate 

    """
    def __init__(self, learning_rate):
        # super(SGD, self).__init__()
        self.learning_rate = learning_rate
    
    def update_layers(self):
        """inherited method from base class
        """
        for layer in self.layers:
            for param in layer.params:
                param.data -= param.grad.data * self.learning_rate
                param.grad.data *= 0

    def get_params(self):
        params = {'learning_rate': float(self.learning_rate)}
        return list(params)

class Momentum(Optimizer):
    """Momentum method helps accelerate SGD relevant direction. It does this by adding a fraction momemtum of the updated vector 
    of the past time step to the current update vector.

    v = -lr * dx + v * momemtum
    """
    def __init__(self, learning_rate=0.1, momemtum=0.0):
        # super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update_layers(self):
        if self.v is None:
            self.v = {} # strore update vals
        for i, layer in enumerate(self.layers):
            self.v[i] = {}
            for j, param in enumerate(layer):
                self.v[i][j] = np.zeros_like(param.data)

        for i, layer in enumerate(self.layers):
            for j, param in enumerate(layer):
                self.v[i][j] = self.momentum * self.v[i][j] + (1 - self.momentum) * param.grad.data
                param.data -= self.learning_rate * self.v[i][j]
				param.grad.data *= 0
                
    def get_params(self):
        params = {'learning_rate': float(self.learning_rate), 
                  'momentum': float(self.momentum)}
        return list(params)
