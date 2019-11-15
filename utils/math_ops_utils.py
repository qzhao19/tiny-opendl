from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def _power(x, num):
    """Compute x to the power p, (x**p).
    Args:
        x: array-like, input value
        num: array_like of ints, exponents

    Returns:
        ndarray or scalar
    """
    return np.power(x, num)


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


def _max(x):
    """Find maximal of array elements and corresponding indices along axis = 1
    Args:
        x: array-like
    Returns:
        max_vals_along_axis and max_indices_along_array
    """
    if len(x.shape) != 2:
        raise ValueError('The size of x shape must be 2 dimension')
    
    max_vals = x.max(1)
    max_inds = x.argmax(1)
    return max_vals, max_inds


def _relu(x):
    """Rectified linear unit.
    Args:
        x: a numpy array
    Returns:
        a numpy array. let each elements in array all greater or equal 0
    """
    return np.where(x>0, x, 0)

def _sigmoid(x):
    """Sigmoid activation function. f(x) = 1/(1 + exp(-x))
    Args:
        x: array-like 
    Returns:
        A array with the same type as `x`
    """
    return 1 / (1 + np.exp(-x))

def _tanh(x):
    """Element-wise tanh. Specific: (exp(x)-exp(-x))/(exp(x)+exp(-x))
    Args:
        x: array-like
        
    Returns:
        An array
    """
    return 1 - (2 / (np.exp(2*x) + 1))

def _softmax(x):
    """Computes softmax activations. softmax = exp(x) / sum(exp(x), axis=-1)
    Args:
        x: array-like
        
    Returns:
        An array
    """
    max_vals = np.max(x, axis=-1, keepdims=True)
    # avoid overflow/underflow issue
    exp_vals = np.exp(x - max_vals)
    sum_vals = np.sum(exp_vals, axis=-1, keepdims=True)
    return exp_vals / sum_vals


def get_dropout_mask(x, rate):
    """Compute dropout

        With probability `rate`, drops elements of `x`. Input that are kept are scaled up by `1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
    the expected sum is unchanged.

    Args:
        x: array-like input
        rate: Float between 0 and 1. the probabilty that each element will be dropped
    
    Returns:
        An array-like of same shape of x
    """

    if not isinstance(rate, float):
        if not (rate >= 0 and rate < 1):
            raise ValueError('Rate must be a float between 0 and 1, got %g' %rate)
    if rate == 0:
        return x

    mask = np.random.random(x.shape) 
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    keep_mask = mask >= rate
    return keep_mask
