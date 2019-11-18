from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Optimizer(object):
    """Optimizer base class

    """
    def __init__(self):
        pass

    def set_layers(slef, layers):
        self.layers = layers
    
    def update_layers(self):
        """Update params in sub-class method 
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    """

    def __init__(self, learning_rate):
        super(SGD, self).__init__()
        self.learning_rate = learning_rate
    
    def update_layer(self):
        """inherited method from base class
        """
        for layer in self.layers:
            for param in layer.params:
                param.data -= param.grad.data * self.learning_rate
				param.grad.data *= 0