import numpy as np


def _add(x, y):
    """Addition math operation
    Args:
        x, y: array_like, the arrays to be added.
    Returns:
        output: ndarray
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')
    
    if x.shape != y.shape:
        raise ValueError('x and y must be the common shape')

    return x + y


def _sub(x, y):
    """Subtraction 
    Args:
        x, y: array_like, the arrays to be subtracted.
    Returns:
        output ndarray
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')
    
    if x.shape != y.shape:
        raise ValueError('x and y must be the common shape')

    return x - y


def _neg(x):
    """Computes numerical negative value element-wise.
    Args:
        x: array_like 
    Returns:
        output negative array
    """
    return x * (-1)


def _mul(x, y):
    """Computes multiplication element-wise.
    Args:
        x, y: array_like data
    Returns:
        ndarray 
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')
    
    return x * y

def _matmul(x, y):
    """Multiplies matrix a by matrix b, producing a * b.
    Args:
        x: Tensor of type float or int and rank > 1
        y: Tensor with sampe type and rank as x
    Returns:
        A `Tensor` of the same type as x and y
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')

    if len(x.shape) < 2 or len(y.shape) < 2:
        raise ValueError('The inputs must be tensors of rank >= 2')

    return x.dot(y)

def _transpose(x, axes):
    """Permute the dimensions of an array.
    Args:
        x: array-like, input array
        axes : list of ints, optional By default, reverse the dimensions, otherwise permute the axes according to the values given.
    Returns:
        p : ndarray x with its axes permuted. 
    """
    return np.transpose(x, axes)

def _sum(x, axis):
    """Sum of array elements over a given axis.
    Args:
        x: array-like
        axis: int optional, None or int or tuple of ints, Axis or axes along which a sum is performed. The default, axis=None, will 
              sum all of the elements of the input array.  If axis is negative it counts from the last to the first axis.
    Returns:
         sum_along_axis : ndarray
    """
    return x.sum(axis)

