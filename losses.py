from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor import Tensor
from .utils.math_ops_utils import *


class Loss(object):
    """Loss base class. To be implemented by sub-classes 'backward'
    """
    def backward(self, y_true, y_prde):
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

    def acc(self, y_true, y_pred):
        pass


class MSE(Loss):
    """Computes the mean of squares of errors between labels and predictions.
    `loss = square(y_true - y_pred)`
    """
    def backward(self, y_true, y_pred):
        """Sub-class of backward of Loss, back propagatation from loss function
        """
        # inhrited from Loss
        super().backward(y_treu, y_pred)
        loss = (y_pred - y_true).__power__(2).sum(0)
        error = loss.data.sum()
        acc = self.acc(y_true, y_pred)

        loss.backward(Tensor(np.ones_like(loss.data)))

        return error, 

    def acc(self, y_true, y_pred):
        labels = np.argmax(y_true, axis=1)
        logits = np.argmax(y_pred, axix=1)

        true_preds =  np.sum(labels == logits)
        return true_pred / len(labels)




