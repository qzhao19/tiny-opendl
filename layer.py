from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from placeholder import Input
import numpy as np


class Layer(object):
    """Base layers class. This is the class from whicn all layers inherit.
    A layer is class implementing common neural network operations, e.g. 
    convolutional layer, recurrent layer, etc.

    Args:
        parameters: list, containing all parameters e.g. weights, biases
        input_tensor: like-tensor, input tensor 
        input_shape: The shape of the tensor

    """
    def __init__(self):
        self.params = []
        self.input_tensor = None
        self.input_shape = None
        self.output_shape = None

    def add_params(self, params):
        """Input params that layer need to add, e.g. weights, biases
        Args:
            parameters: list, layer params 
        """
        params.set_trainable(trainable=True)
        self.params.append(params)

    def get_params(self):
        """get all params that the layer need to update 
        """
        return self.params

    def set_input_shape(self, input_shape):
        """setup the input tensor shape for layer
        Args:
            input_shape: int list, input tensor shape
        """
        self.input_shape = input_shape
    
    def get_input_shape(self):
        """get input tensor shape
        """
        return self.input_shape[0]

    def set_output_shape(self, output_shape):
        """setup output tensor shape for layers
        Args:
            output_shape: int list, output shape
        """
        self.output_shape = output_shape

    def get_output_shape(self):
        """get output tensor shape, it will be realized from sub-class
        """
        raise NotImplementedError('Not implemented sub-class method: get_output_shape')
    
    def init_params(self):
        """initialize layer parameters
        """
        # raise NotImplementedError('Not implemented sub-class method: init_params')
        pass

    def __call__(self, input_pl):
        """callable object, a operator for model
        Args:
            input_pl: input placeholder
        """
        if not isinstance(input_pl, Input):
            raise ValueError('Input placeholder mus be same type with Input')

        # setup input_placeholder as current layer input, get shape
        self.set_input_shape(input_pl.shape)
        # get output_shape = input_pl_shape,  takes input_pl_shape as output shape of a layer
        output_shape = self.get_output_shape()
        # create a output_placeholder
        return Input(output_shape, [input_pl], self)


class Add(Layer):
    """Layer that adds a list of inputs. It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).
    """
    def __init__(self):
        super(Add, self).__init__()

    def get_output_shape(self):
        """inherited class from Layer to compute output shape
        """
        # input shape of a layer is equal to output shape
        output_shape = self.get_input_shape()
        # setup output shape
        self.set_output_shape(output_shape)
        return output_shape
    
    def forward(self, input_tensor1, input_tensor2):
        return input_tensor1.__add__(input_tensor2)

    def __call__(self, input_pl1, input_pl2):
        """callable method to add 2 layer
        """
        if (not isinstance(input_pl1, Input)) and (not isinstance(input_pl2, Input)):
            raise ValueError('Layer1 and layer2 must be same type')
        
        if input_pl1.shape != input_pl2.shape:
            raise ValueError('Layer1 and layer2 must be same shape')

        # input_pl is current layer input
        self.set_input_shape(input_pl1.shape)
        output_shape = self.get_output_shape()

        # create output placeholder
        return Input(output_shape, [input_pl1, input_pl2], self)



class Flatten(Layer):
    """Flattens the input. Does not affect the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def get_output_shape(self):
        input_shape = self.get_input_shape()
        # output_shape = (batch_size, channel*height*width)
        output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.set_output_shape(output_shape)
        return output_shape
    def forword(self, input_tensor):
        return input_tensor.flatten()



class Reshape(Layer):
    """Reshapes an output to a certain shape.

    Args:
        target_shape: integers, Target shape. 
    """
    def __init__(self, target_shape):
        self.target_shape = target_shape
        super(Reshape, self).__init__()

    def get_output_shape(self):
        output_shape = target_shape
        self.set_output_shape(output_shape)
        return output_shape
    
    def forward(self, input_tensor):
        return input_tensor.reshape(target_shape)


class Activation(Layer):
    """Applies an activation function to an output.
    Activation subclass inherts from base class Layer

    Args:
        activation: Activation function, string name of built-in activation function, such as "relu".

    """
    def __init__(self, activation):
        # In Python 3, we could use super()__init__() syntaxe to inherts baseclass 
        super(Activation, self).__init__()
        self.activation = activation

    def get_output_shape(self):
        """This method is inherited from base-class
        """
        self.set_output_shape(self.get_input_shape())
        return self.output_shape

    def forward(self, input_tensor):
        """forward propagation function
        Args:
            input_tensor: Tensor, input tensor fo shape [batch_size, c, h, w]
        """
        if self.activation == 'relu':
            return input_tensor.relu()
        elif self.activation == 'sigmoid':
            return input_tensor.sigmoid()
        elif self.activation == 'tanh':
            return input_tensor.tanh()
        elif self.activation == 'softmax':
            return input_tensor.softmax()
        else:
            raise ValueError('There are not %s activation function, default activations are relu, sigmoid, tanh and softmax')


class Dropout(Layer):
    """Applies Dropout to the input.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
    """
    def __init__(self, rate):
        self.rate = rate
        super(Dropout, self).__init__()

    def get_output_shape():
        output_shape = self.get_input_shape()
        self.set_output_shape(output_shape)
        return output_shape
    def forward(self, input_tensor):
        return input_tensor.dropout(self.rate)

