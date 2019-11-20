from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from inputs import Input


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
        raise NotImplementedError('Not implemented sub-class method: init_params')

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
        output_shape = self.target_shape
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
        output_shape = self.get_input_shape()
        self.set_output_shape(output_shape)
        return output_shape

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
        super(Dropout, self).__init__()
        self.rate = rate
        
    def get_output_shape():
        output_shape = self.get_input_shape()
        self.set_output_shape(output_shape)
        return output_shape

    def forward(self, input_tensor):
        return input_tensor.dropout(self.rate)


class Conv2D(Layer):
    """2D convolution layer
    Args:
        filter_nums: Integer, the number of filters in the convolution
        ksize: Tuple or list of integer, the length of the convolution window.
        stride: Integer, the stride length of the convolution.
        pad: Integer, the padding size
        input_shape: Tuple of list of integer, the input tensor shape, default value is None
        weight_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.

    """
    def __init__(self, filter_nums, ksize, stride=1, pad=0, input_shape=None, weight_initializer='normal', bias_initializer='zeros'):
        # inherite base class from Layer
        super(Conv2D, self).__init__()
        self.filter_nums = filter_nums
        self.ksize = ksize
        self.stride =stride
        self.pad = pad

        if input_shape is not None:
            input_c, input_h, input_w = input_shape
            self.set_input_shape((None, input_c, input_h, input_w))

        # self.weight_vals = initializer.get(weight_initializer)
        # self.bias_vals = initializer.get(weight_initializer)
        self.weight = None
        self.bias = None

    def get_output_shape(self):
        """Should to compute output conv layer shape
        """
        input_nums, input_c, input_h, input_w = self.get_input_shape()
        # comput output_h and output_w from function get_conv_output_shape
        output_h, output_w = get_conv_output_shape(input_h, input_w, self.ksize, self.stride, self.pad)
        output_shape = (input_nums, self.filter_nums, output_h, output_w)
        self.set_output_shape(output_shape)
        return output_shape

    def init_params(self):
        """sub-class method that inherit from base class Layer
        """
        kernel_h, kernel_w = self.ksize
        input_c = self.get_input_shape()[1]
        # weight_vals = initializer.get(weight_initializer)
        # bias_vals = initializer.get(bias_initializer)

        weight_vals = np.random.randn(input_c * kernel_h * kernel_w, self.filter_nums) * np.sqrt(2.0 / input_c * kernel_h * kernel_w)
        bias_vals = np.zeros(self.filter_nums)

        self.weight = Tensor(weight_vals, auto_grad=True)
        self.bias = Tensor(bias_vals, auto_grad=True)
        # apend weight and bias into self.params
        self.params.append(self.weight)
        self.params.append(self.bias)

    def forward(self, input_tensor):
        """forward propagation 
        Args:
            input_tensor: Tensor with shape of [batch_size, c, h, w]

        Returns:
            output tensor of shape [batch_size, c, output_h, output_w]
        """
        input_nums, input_c, input_h, input_w = input_tensor.shape
        output_h, output_w = get_conv_output_shape(input_h, input_w, self.ksize, self.stride, self.pad)
        # determine weight and bias are existed
        if (self.weight is None) or (self.bias is None):
            self.get_input_shape(input_nums, input_c, input_h, input_w)
            self.get_output_shape()
            self.init_params()

        # Expand input data into a two-dimensional array of shape [input_nums*output_h*output_w, input_c*kernel_h*kernel_w]
        matrix = input_tensor.tensor_to_matrix(self.ksize, self.stride, self.pad)
        # shape [input_num*output_h*output_h, filter_nums]
        output = matrix.dot(self.weight)
        output = output + self.bias.expand(0, matrix.data.shape[0]) # [input_nums * output_h * output_w, filter_nums]
        return output.reshape(input_nums, output_h, ouput_w, -1).transpose(0, 3, 1, 2) # [input_nums, filter_nums, output_h, output_w]


class MaxPooling(Layer):
    """Max pooling operation for spatial data.

    Attributs:
        pool_size: Integer, size of the max pooling windows.
        strides: Integer, Factor by which to downscale.
        pad: Integer, the padding size
        input_shape: shape tupel or list


    """
    def __init__(self, pool_size, stride=1, pad=0, input_shape=None):
        super(MaxPooling, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.input_shape = input_shape

        if input_shape is not None:
            input_c, input_h, input_w = input_shape
            self.set_input_shape((None, input_c, input_h, input_w))

    def get_output_shape(self):
        """Should to compute output maxpooling layer shape
        """
        input_nums, input_c, input_h, input_w = self.get_input_shape()
        # comput output_h and output_w from function get_conv_output_shape
        output_h, output_w = get_conv_output_shape(input_h, input_w, self.pool_size, self.stride, self.pad)
        output_shape = (input_nums, self.filter_nums, output_h, output_w)
        self.set_output_shape(output_shape)
        return output_shape

    def forward(self, input_tensor):
        """forward propagation 
        Args:
            input_tensor: Tensor with shape of [batch_size, c, h, w]

        """
        # setup input_tensor shape
        self.set_input_shape(input_tensor.data.shape)
        self.get_output_shape()


        input_nums, input_c, input_h, input_w = input_tensor.data.shape
        # compute output h and w from the function get_conv_output_shape
        output_h, output_w = get_conv_output_shape(input_h, input_w, self.pool_size, self.stride, self.pad)

        # Expand input data into a two-dimensional array of shape [input_nums*output_h*output_w, input_c*kernel_h*kernel_w]
        matrix = input_tensor.tensor_to_matrix(self.pool_size, self.stride, self.pad)
        matrix = matrix.reshape((-1, self.pool_size[0] * self.pool_size[1]))
        output = matrix.max()
        return output.reshape((input_nums, output_h, output_w, input_c)).transpose((0, 3, 1, 2))


class Dense(Layer):
    """Regular densely-connected NN layer.

    Dense implements the operation: output = dot(inputs, weight) + bias

    Args:
        output_dim: Integer, dim of output = input_shape[-1] 
        input_shape: shape tupel or list
        weight_initializer: String 
        bias_initializer: String

    """
    def __init__(self, output_dim, input_shape=None, weight_initializer='normal', bias_initializer='zeros'):
        super(Dense, self).__init__()
        self.output_dim = output_dim
        self.weight = None
        self.bias = None

        if input_shape is not None:
            self.set_input_shape((None,) + input_shape)

    def get_output_shape(self):
        input_shape = self.get_input_shape()
        output_shape = input_shape[:-1] + (self.output_dim, )
        self.set_output_shape(output_shape)
        return output_shape

    def init_params(self):
        # get input and output dim
        input_dim = self.get_input_shape()[-1]
        output_dim = self.get_output_shape()[-1]

        # init weight and bias
        # weight_vals = initializer.get(weight_initializer)
        # bais_vals = initializer(bias_initializer)
        weight_vals = np.random.random(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        bias_vals = np.zeros((output_dim))

        self.weight = Tensor(weight_vals, auto_grad=True)
        self.bais = Tensor(bias_vals, auto_grad=True)
        # add params into param_list
        self.add_params(weight)
        self.add_params(bias)

    def forward(self, input_tensor):
        """forward compute fot sequence and graph model
        """
        if self.weight is None:
            self.set_input_shape(input_tensor.data.shape)
            output_shape = self.get_output_shape()
            self.init_params()

        if len(input_tensor.data.shape) == 2:
            batch_size, _ = input_tensor.data.shape
            # kernel = dot(inputs, weights)
            output = input_tensor.data.dot(self.weight) + self.bias.expand(0, batch_size)
            return output
        elif len(input_tensor.data.shape) == 3:
            batch_size, repeats, _ = input_tensor.data.shape
            kernel = input_tensor.data.dot(self.weight)
            output = kernel + self.bias.expand(0, repeats).expand(0, batch_size)
            return output

