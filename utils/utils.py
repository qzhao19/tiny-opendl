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
    if len(inputs.shape) != 4:
        raise ValueError('The shape of input tensor must be [inputs_nums, channel, height, width]')
    
    if len(ksize) != 2:
        raise ValueError('Kernel size must be a list of shape [kernel_h, kernel_w]')

    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]

    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    inputs_nums, inputs_c, inputs_h, inputs_w = inputs.shape
    kernel_h, kernel_w = ksize
    # calculate output shape: output_h and output_w
    output_h = (inputs_h + pad*2 - kernel_h) // stride + 1
    output_w = (inputs_w + pad*2 - kernel_w) // stride + 1

    # define tensor and matrix
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
    You can think of this method as the reverse function of tensor_to_matrix. It is adding up the corresponing indices and transforms the given matrix
    back into the initial shape.

    Assuming you have a matrix_to_tensor transformed matrix with shape (9,600*13*13). The original image had a shape of (600,1,28,28) with padding P = 0 
    and stride S = 2. col2im creates out of the im2col matrix and the same hyperparameter a new matrix with a shape of (600, 1, 28, 28).

    Args:
        inputs: 2D matrix of shape [batch_size*output_h*output_w, batch_size*kernel_h*kernel_w] 
        shape: int list, tensor of shape [batch_size, channel, height, width]
        ksize: int list, filter shape of [kernel_h, kernel_w] 
        stride: int, the filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding pads the input volume with zeros around the border, default value 0
    Returns:
        4D tensor of shape [batch_size, channel, height, width]
    """
    if len(inputs.shape) != 4:
        raise ValueError('The shape of input tensor must be [inputs_nums, channel, height, width]')

    if len(ksize) != 2:
        raise ValueError('Kernel size must be a list of shape [kernel_h, kernel_w]')
    
    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]

    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

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


def get_conv_output_shape(inputs_h, inputs_w, ksize, stride=1, pad=0):
    """compute convolution's shape
    Args:
        inputs_h: int, inputs tensor height
        inputs_w: int, inputs width
        ksize: int list, filter shape of [kernel_h, kernel_w]
        stride: int, the filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding numbers
    Returns:
        outputs height and output width
    """

    if len(ksize) != 2:
        raise ValueError('Kernel size must be a list of shape [kernel_h, kernel_w]')

    if not isinstance(inputs_h, int):
        inputs_h = int(inputs_h)
    
    if not isinstance(inputs_w, int):
        inputs_w = int(inputs_w)
    
    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]
    
    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    kernel_h, kernel_w = ksize
    # calculate output height/width
    output_h = int((inputs_h + pad*2 - kernel_h) / stride) + 1
    output_w = int((inputs_w + pad*2 - kernel_w) / stride) + 1
    return (output_h, output_w)

def one_hot_encoder(y, label_nums=None):
    """One hot encoding method, Categorical data must be converted be number data.A one hot encoding is a representation of categorical variables as binary vectors.
    data, This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all 
    zero values except the index of the integer, which is marked with a 1.
    Args:
        y: predicting target 
        label_nums: the number of labels
    Returns:
        one hot encoded label
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if len(y.shape) != 1:
        raise ValueError('The size of input shape must be 1!')

    if label_nums is not None:
        label_nums = label_nums
    else:
        label_nums = np.max(y) + 1
    
    one_hot = np.zeros((y.shape[0], label_nums), dtype=y.dtype)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def get_batch_data(data, batch_size=10):
    """get dataset by one batch 
    Args:
        data: numpy.ndarray or list type data
        batch_size: int, the batch_size
    Returns:
        one batch dataset
    """
    if isinstance(data, list):
        sample_nums = len(data.shape[0])
    else:
        sample_nums = data.shape[0]
    # get the number of sample 
    sample_nums = data.shape[0]
    for i in np.arange(0, sample_nums, batch_size):
        # define begin and end idx
        begin, end = i, min(i+batch_size, sample_nums)
        if isinstance(data, list):
            yield tuple([x[begin:end] for x in data])
        else:
            yield data[begin:end]


def shuffle_data(X, y, seed=None):
    """Randomly shuffle samples of X and y
    Args:
        X: data
        y: label
        seed: int, the number used to initialize pseudorandom number generator
    Returns:
        X, y
    """
    if seed:
        np.random.seed(seed)
    # define indice of elements
    indices = np.arange(X.shape[0])
    # shuffle
    np.random.shuffle(indices)
    return X[indices], y[indices]



def split_train_test(X, y, ratio=0.7, shuffle=True, seed=None):
    """Split dataset into training data and testing data
    Args:
        X: data
        y: label
        ration, float, test ratio, default value is 0.6
        shuffle: boolean, shuffle data or not
        seed: int, the number used to init random shffle
    Returns:
        training data and testing data: (train_X, train_y), (test_X, test_y)
    """
    # if shuffle dataset
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    
    indices = int(X.shape[0]*ratio)
    train_X, train_y = X[:indices], y[:indices]
    test_X, test_y = X[indices:], y[indices:]

    return (train_X, train_y), (test_X, test_y)

def select_array_indice(array, dim, i):
    """Select subarray data of some dimension according to indices given
    Args:
        array: array data
        dim: some dimenion
        i: indice given
    returns:

    """
    indices = [slice(None)] * array.ndim
    indices[dim] = i
    # print(indices)
    return array[tuple(indices)]

def plus_array_indice(value, array, dim, i):
    """plus element into selected subarray data from an indices given
    Args:
        value: elments adding to subarray
        array: array data
        dim: certain dimension
        i: indices given
    """
    indices = [slice(None)] * array.ndim
    indices[dim] = i
    array[tuple(indices)] += value








