from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensors import Tensor
from .utils.math_ops_utils import *


class Loss(object):
    """Loss base class. To be implemented by sub-classes 'backward'
    """
    def backward(self, y_true, y_pred):
        """Back propagatation from loss function

        Args:
            y_true: Tensor, ground truth vals 
            y_pred: Tensor, predict vals
        """
        # check y_true and y_pred type
        if not isinstance(y_true, Tensor):
            raise ValueError('The ground truth values must be Tensor')
        if not isinstance(y_pred, Tensor):
            raise ValueError('The prediction must be Tensor')
        if y_true.data.shape != y_pred.data.shape:
            raise ValueError('The y_true should be same shape that y_pred')

    def compute_acc(self, y_true, y_pred):
        y_true_arg = np.argmax(y_true, axis=1)
        y_pred_arg = np.argmax(y_pred, axis=1)

        # true_pred_arg =  np.sum(y_true_arg == y_pred_arg)
        return np.sum(y_true_arg == y_pred_arg) / len(y_true_arg)


class MSE(Loss):
    """Computes the mean of squares of errors between labels and predictions.
    `loss = square(y_true - y_pred)`
    """
    def backward(self, y_true, y_pred):
        """Sub-class of backward of Loss, back propagatation from loss function
        Args:
            y_true: Tensor, labels
            y_pred: Tensor, logits
        """
        # inhrited from Loss
        super().backward(y_true, y_pred)
        loss = (y_pred - y_true).__power__(2).sum(0)
        error = loss.data.sum()

        acc = self.compute_acc(y_true.data, y_pred.data)

        loss.backward(Tensor(np.ones_like(loss.data)))

        return error, acc


class SoftmaxCrossEntropy(Loss):
    """Computes softmax cross entropy between `logits` and `labels`.
        
    Measures the probability error in discrete classification tasks in which the
    classes are mutually exclusive (each entry is in exactly one class).
    """
    def backward(self, y_true, y_pred):
        """softmax cross entropy
        """
        super().backward(y_true, y_pred)
        # transform logtis vals into probabilities
        logits = softmax(y_pred.data)

        loss = y_pred.cross_entropy(y_true)
        error = loss.data.sum()

        acc = self.compute_acc(y_true.data, y_pred.data)

        loss.backward(Tensor(np.ones_like(loss.data)))

        return error, acc

