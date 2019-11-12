from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

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

