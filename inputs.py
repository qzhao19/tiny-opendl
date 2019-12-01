from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Input(object):
    """A placeholder class for a tensor that be fed. Layer to be used as an entry point into a Network.
    This class can create placeholders for Tensors
    
    Args:
        shape: int list, the shape of the tensor to be fed, 
               A shape tuple, not including the batch size.
        input_pl: the input placeholder to take the last layer placeholder
        input_layer: tensor, input tensor
    
    Methods:
        set_name: 
        set_tensor: 
        set_shape:
        set_input_placeholder:
        set_input_layer:
    """
    def __init__(self, shape=None, input_pl=None, input_layer=None):
        # An optional name string for the layer.
        self.id = None
        self.tensor = None
        if shape is None and input_pl is None:
            raise ValueError('Provide the input shape and input placeholder')
        self.shape = shape
        self.input_pl = input_pl
        self.input_layer = input_layer

    def set_id(self, id):
        self.id = id
    
    def set_tensor(self, tensor):
        self.tensor = tensor
    
    def set_shape(self, shape):
        self.shape = shape
    
    def set_input_pl(self, input_pl):
        self.input_pl = input_pl
    
    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
