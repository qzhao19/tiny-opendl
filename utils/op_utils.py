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
        raise TypeError('x and y must be the common shape')

    return x + y


def _subtract(x, y):
    """Subtraction 
    Args:
        x, y: array_like, the arrays to be subtracted.
    Returns:
        output ndarray
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')
    
    if x.shape != y.shape:
        raise TypeError('x and y must be the common shape')

    return x - y


def _negative(x):
    """Computes numerical negative value element-wise.
    Args:
        x: array_like 
    Returns:
        output negative array
    """
    return x * (-1)


def _multiply(x, y):
    """Computes multiplication element-wise.
    Args:
        x, y: array_like data
    Returns:
        ndarray 
    """
    if x.dtype is not y.dtype:
        raise TypeError('x and y should be same type.')
    
    return x * y
