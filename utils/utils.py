import numpy as np
import matplotlib.pyplot as plt

class Util:
    """get only one random number
    """
    random_sets = {}
    # here, we use @staticmethod to recall methods in class Util
    @staticmethod
    def unique_id(group='default'):
        """This function is to find a random tensor id 
        Args:
            group: string, default value 'default'
        """
        # inti random id
        random_id = np.random.randint(0, 1000000)
        # make sure ranodm_id is not in random_sets
        if group not in Util.random_sets:
            Util.random_sets[group] = set([random_id])
            return random_id
        # make loop to find one random_id non repeated
        while random_id in Util.random_sets[group]:
            random_id = np.random.randint(0, 1000000)
        # add found random_id into set
        Util.random_sets[group].add(random_id)
        return random_id

    @staticmethod
    def clear():
        """clear random set
        """
        Util.random_sets = {}

def tensor_to_matrix(inputs, ksize, stride=1, pad=0):
    """Reshape tensor of shape [batch_size, channel, height, width] into matrix with shape [batch_size*output_h*output_w, channel*kernel_h*kernel_w] 
   
    Assume you have a image of shape (600, 1, 28, 28), padding=0, stride=2 and a filter with dimensions (3,3). You already know 
    that the output dimension of a convolution operator has to be (13,13) with (28-3)/2 + 1 = 13. tensor_to_matrix creates then a new matrix 
    with the shape of (9 * 1, 600 * 13 * 13) which you then can matrix multiply with your flattend kernel of shape 
    (n,9 * 1). The multiplication will result into a new matrix of shape (n,600*13*13) which you can then reshape into your convolution 
    output (600, n, 13, 13) which is the wanted result. Note that n is the numbers of filters inside your convolution layer.

    Args:
        inputs: 4D inputs tensor of shape [batch_size, channel, height, width]
        ksize: int list, filter shape of [kernel_h, kernel_w] 
        stride: int, he filter convolves around the input volume by shifting one unit at a time
        pad: int, zero padding pads the input volume with zeros around the border
    Returns:
        2D matrix with same type data that tensor of shape [batch_size*output_h*output_w, channel*kernel_h*kernel_w] 
    """
    inputs_nums, inputs_c, inputs_h, inputs_w = inputs.shape
    kernel_h, kernel_w = ksize
    # calculate output shape: output_h and output_w
    output_h = (inputs_h + pad*2 - kernel_h) // stride + 1
    output_w = (inputs_w + pad*2 - kernel_w) // stride + 1

    # define tensor and its return
    tensor = np.pad(inputs, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    matrix = np.zeros((inputs_nums, inputs_c, kernel_h, kernel_w, output_h, output_w), dtype=inputs.dtype)

    for y in range(kernel_h):
        y_max = y + output_h * stride
        for x in range(kernel_w):
            x_max = x + output_w * stride
            matrix[:, :, y, x, :, :] = tensor[:, :, y:y_max:stride, x:x_max:stride]

    matrix = matrix.transpose(0, 4, 5, 1, 2, 3).reshape(inputs_nums*output_h*output_w, -1)
    return matrix


def matrix_to_tensor(inputs, shape, ksize, stride=1, pad=0):
    """Reshape matrix of shape  [batch_size*out_h, batch_size*out_w] into tensor with shape [batch_size, channel, height, width] 
    You can think of this method as the reverse function of im2col. It is adding up the corresponing indices and transforms the given matrix
    back into the initial shape.

    Assuming you have a im2col transformed matrix with shape (9,600*13*13). The original image had a shape of (600,1,28,28) with padding P = 0 
    and stride S = 2. col2im creates out of the im2col matrix and the same hyperparameter a new matrix with a shape of (600, 1, 28, 28).

    Args:
        inputs: 2D matrix of shape [batch_size*output_h*output_w, batch_size*kernel_h*kernel_w] 
        shape: int list, tensor of shape [batch_size, channel, height, width]
        ksize: int list, filter shape of [kernel_h, kernel_w] 
        stride: int, he filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding pads the input volume with zeros around the border, default value 0
    Returns:
        4D tensor of shape [batch_size, channel, height, width]
    """
    # get inputs 4d tensor shape
    inputs_nums, inputs_c, inputs_h, inputs_w = shape
    kernel_h, kernel_w = ksize
    # calculate output tensor shape
    output_h = (inputs_h + pad*2 - kernel_h) // stride + 1
    output_w = (inputs_w + pad*2 - kernel_w) // stride + 1

    # here, we difine a matrix with shape [nums, out_height, out_width, in_channel, kernel_height, kernel_width]
    matrix = inputs.reshape(inputs_nums, output_h, output_w, inputs_c, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    tensor = np.zeros((inputs_nums, inputs_c, inputs_h + pad*2 + stride - 1, inputs_w + pad*2 + stride - 1), dtype=inputs.dtype)

    for y in range(kernel_h):
        y_max = y + output_h * stride
        for x in range(kernel_w):
            x_max = x + output_w * stride
            tensor[:, :, y:y_max:stride, x:x_max:stride] += matrix[:, :, y, x, :, :]

    return tensor[:, :, pad:inputs_h + pad, pad:inputs_w + pad]






