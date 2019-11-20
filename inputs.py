from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Input(object):
    """A placeholder class for a tensor that be fed 
    
    Args:
        shape: int list, the shape of the tensor to be fed
        input_pl: the placeholder to be input
        input_layer: tensor, input layer
    
    Methods:
        set_name: 
        set_tensor: 
        set_shape:
        set_input_placeholder:
        set_input_layer:
    """
    def __init__(self, shape=None, input_pl=None, input_layer=None):
        self.name = None
        # self.tensor = None
        if shape is None and input_pl is None:
            raise ValueError('Provide the input shape and input placeholder')
        self.shape = shape
        self.input_pl = input_pl
        self.input_layer = input_layer

    def set_name(self, name):
        self.name = name
    
    # def set_tensor(self, tensor):
    #     self.tensor = tensor
    
    def set_shape(self, shape):
        self.shape = shape
    
    def set_input_pl(self, input_pl):
        self.input_pl = input_pl
    
    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
