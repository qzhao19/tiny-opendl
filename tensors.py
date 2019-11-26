from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .utils.array_ops_utils import *
from .utils.math_ops_utils import *
import numpy as np

class Tensor(object):
    def __init__(self, data, parents=None, op=None, auto_grad=None, id=None):
        """Initialize tensor
        Args:
            data: ndarray, input tensor data
            parents: list, source tensor node to generate new tensor
            op: str, tensor operation key work, i.g. add, subtract, multiply, dot etc..
            auto_grad: str, automatic gradient keyword and check actual node whether could backpropagate
            id: tensor node id that could be generate by function unique_id from tensor_utils.py

        """
        # all data MUST BE numpy ndarray type data
        self.data = np.array(data)
        self.parents = parents
        # define operations on tensor
        self.op = op
        self.op_params = None
        # define automatic gradient and check actual node whether could backpropagate
        self.auto_grad = auto_grad
        self.trainable = False
        # create node unique id
        if id is None:
            id = unique_id('Tensor')
        self.id = id

        # init gradient of node
        self.grad = None
        self.momentum = None

        # define a hashmap as childrens node of parents node
        # it contains child_id and child_cnt: the number of count about child node appeared
        self.children = {}

        # create parents and children of tensor dependency relation
        # first, parents node must be not None, make sure that children node should be existed
        # increments children node, one parent could have plural children
        if parents is None:
            return
        for parent in parents:
            if self.id not in parent.children:
                parent.children[self.id] = 0
            parent.children[self.id] += 1

    def set_trainable(self, trainable):
        self.trainable = trainable
    

    def has_received_all_children_grads(self):
        """check actual node whether has received gradients from all children node 
        """
        # children: child_id + child_cnt
        for child_id, child_cnt in self.children.items():
            if child_cnt > 0:
                # if child_cnt sup 0, which means receive children grads
                return False
        return True

    def backward(self, child_grad=None, child_node=None):
        """back propagatation function
        Args:
            child_gard: gradient of child node
            child_node: child node

        """
        # 1. make sure if actual node could be backprobagate, before, have defined auto_grad
        # if not, class object will be deleted  
        if not self.auto_grad:
            del self
            return 
        
        # actual node received gradients from childs nodes
        # make sure that child node grad and actual node grad are same type data
        assert isinstance(child_grad, Tensor)
        # if gradient is none, create a new tensor as gradient
        if self.grad is None:
            self.grad = Tensor(child_grad.data)
        else:
            self.grad.data += child_grad.data
        # decrements child node id, if child_id is not None
        if child_node is not None:
            self.children[child_node.id] -= 1
            # make sure the number of children nodes is not None
            assert self.children[child_node.id] >= 0
        # if parents nodes doensn exists, need not to backpropagate to child node
        if self.parents is None:
            return 
        # if child_node is existed, and actual node has not received gradients from children node
        if child_node is not None and not self.has_received_all_children_grads():
            return 
        
        # 2. actual node backpropagate to parents nodes
        # define gradient operator for each math ops
        if self.op == 'add':
            grad = Tensor(self.grad.data)
            self.parents[0].backward(grad, self)
            self.parents[1].backward(grad, self)

        elif self.op == 'sub':
            grad1 = Tensor(self.grad.data)
            grad2 = Tensor(-self.grad.data)
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)
        
        elif self.op == 'neg':
            grad = Tensor(-self.grad.data)
            self.parents[0].backward(grad, self)
        
        # multiplication of scalar
        elif self.op == 'mul':
            grad1 = Tensor(self.grad.data * self.parents[1].data)
            grad2 = Tensor(self.grad.data * self.parents[0].data)
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)
        
        # multiplication of tensor
        elif self.op == 'matmul':
            grad1 = Tensor(self.grad.data.dot(self.parents[1].data.transpose()))

            if len(self.grad.data.shape) == 2:
                grad2 = Tensor(self.parents[0].data.transpose().dot(self.grad.data))

            elif len(self.grad.data.shape) == 3:
                batch_size, _, input_dims = self.parents[0].data.shape
                _, _, output_dims = self.grad.data.shape
                # here, new_orders = [-1, 0, 1], which presents [channel, height, width]
                input_data = self.parents[0].data.transpose(-1, 0, 1).reshape(input_dims, -1)
                grad_data = self.grad.data.reshape((-1, output_dims))
                grad2 = Tensor(input_data.dot(grad_data))
            else:
                raise TypeError('Input tensor dimension should be 2 or 3!')
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)


        elif self.op == 'power':
            num = self.op_params['num']
            grad_data = self.grad.data * num * np.power(self.parents[0].data, num-1)
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)


        elif self.op == 'sum':
            axis = self.op_params['axis']
            grad = Tensor(self.parents[0].data.sum(axis))
            self.parents[0].backward(grad, self)


        elif self.op == 'max':
            arg_max = self.op_params['arg_max']
            shape = self.parents[0].data.shape
            grad_data = np.zeros(shape)
            # grad_data[np.arange(shape[0]), arg_max] = self.grad.data
            grad_data[:, arg_max] = self.grad.data
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)


        elif self.op == 'reshape':
            shape = self.parents[0].data.shape
            grad = Tensor(self.grad.data.reshape(shape))
            self.parents[0].backward(grad, self)
        
        elif self.op == 'expand':
            axis = self.op_params['axis']
            repeats = self.parents[0].data.shape[axis]
            expanded_vals = expand_data(self.grad.data, axis, repeats)
            grad = Tensor(expanded_vals)
            self.parents[0].backward(grad, self)
        
        elif self.op == 'transpose':
            if self.op_params['axes'] is not None:
                axes = self.op_params['axes']
                revert_axes = tuple([axes.index(i) for i in range(len(axes))])
                grad = Tensor(self.grad.data.transpose(revert_axes))
            else:
                grad = Tensor(self.grad.data.transpose())
            self.parents[0].backward(grad, self)
            
        elif self.op == 'flatten':
            grad = Tensor(self.grad.data.reshape(self.parents[0].data.shape))
            self.parents[0].backward(grad, self)


        elif self.op == 'concatenate':
            axis = self.op_params['axis']
            lens = [parent.data.shape[axis] for parent in self.parents]
            idxs = np.cumsum(lens)
            
            for i in range(len(self.parents)):
                select_range = range(0, idxs[i]) if i == 0 else range(idxs[i-1], idxs[i])
                grad_data = array_index_select(self.grad.data, axis, select_range)
                self.parents[i].backward(Tensor(grad_data), self)

        # tensor_to_matrix equal to im2col
        elif self.op == 'tensor_to_matrix':
            inputs = self.grad.data
            shape = self.parents[0].data.shape
            ksize = self.op_params['ksize']
            stride = self.op_params['stride']
            pad = self.op_params['pad']
            grad_data = matrix_to_tensor(inputs, shape, ksize, stride, pad)
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)

        elif self.op == 'select_array_indice':
            grad_data = np.zeros_like(self.parents[0].data).astype('float64')
            plus_array_indice(self.grad.data, grad_data, self.op_params['axis'], self.op_params['i'])
            self.parents[0].backward(Tensor(grad_data), self)

        # define gradient operation for activiation function:
        # relu: f'(x) = {0, x<0; 1, x>0}
        elif self.op == 'relu':
            if self.parents[0].data > 0:
                # grad_data = self.grad.data * (self.parents[0].data * np.ones_like(self.parents[0].data))
                grad_data = self.grad.data * (self.data * np.ones_like(self.data))
            else:
                # grad_data = self.grad.data * (self.parents[0].data * np.zeros_like(self.parents[0].data))
                grad_data = self.grad.data * (self.data * np.zeros_like(self.data))
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)
        
        # sigmoid: f'(x) = f(x)*(1-f(x))
        elif self.op == 'sigmoid':
            grad_data = self.grad.data * self.data * (1 - self.data)
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)

        # tanh: f'(x) = 1 - (f(x))²
        elif self.op == 'tanh':
            grad_data = self.grad.data * (1 - np.power(self.data, 2))
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)

        # softmax: f'(x) = (1(i==j)-f(x))
        elif self.op == 'softmax':
            grad_data = self.data * (1 - self.data)
            grad = Tensor(grad_data)
            self.parents[0].backward(grad, self)

        elif self.op == 'cross_entropy':
            logits = self.op_params['logits']
            # limit logit values range
            logits = np.clip(logits, 1e-15, 1 - 1e-15)
            labels = self.parents[1].data
            grad_data = (logits - labels) / logits.shape[0]
            self.parents[0].backward(Tensor(grad_data), self)

        elif self.op == 'dropout':
            grad_data = self.grad.data * self.op_params['mask']
            self.parents[0].backward(Tensor(grad_data), self)

        if not self.trainable:
            del self


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

    def __add__(self, other):
        add_vals = add(self.data, other.data)
        output = Tensor(add_vals, parents=[self, other], op='add', auto_grad=True)
        return output

    def __sub__(self, other):
        sub_vals = sub(self.data, other.data)
        output =  Tensor(sub_vals, parents=[self, other], op='sub', auto_grad=True)
        return output

    def __neg__(self):
        neg_vals = neg(self.data)
        output = Tensor(neg_vals, parents=[self], op='neg', auto_grad=True)
        return output

    def __mul__(self, other):
        mul_vals = mul(self.data, other.data)
        output = Tensor(mul_vals, parents=[self, other], op='mul', auto_grad=True)
        return output

    def __matmul__(self, other):
        matmul_vals = matmul(self.data, other.data)
        return Tensor(matmul_vals, parents=[self, other], op='matmul', auto_grad=True)

    def __power__(self, num):
        power_vals = power(self.data, num)
        output = Tensor(power_vals, parents=[self], op='power', auto_grad=True)
        output.op_params = {'num': num}
        return output

    def sum(self, axis=None):
        sum_vals = sum(self.data, axis)
        output = Tensor(sum_vals, parents=[self], op='sum', auto_grad=True)
        output.op_params = {'axis': axis}
        return output

    def max(self):
        """Find maximal of array elements and corresponding indices along axis = 1
        """
        max_vals, max_inds = max(self.data)
        output = Tensor(max_vals, parents=[self], op='max', auto_grad=True)
        output.op_params = {'argmax': max_inds}
        return output

    def reshape(self, new_shape):
        shaped_vals = reshape(self.data, new_shape)
        output = Tensor(shaped_vals, parents=[self], op='reshape', auto_grad=True)
        # output.op_params = {'new_shape': new_shape}
        return output

    def expand(self, axis, repeats):
        expanded_vals = expand_data(self.data, axis, repeats)
        output = Tensor(expanded_vals, parents=[self], op='expand', auto_grad=True)
        output.op_params = {'axis': axis}
        return output

    def transpose(self, axes=None):
        trans_vals = transpose(self.data, axes=axes)
        output = Tensor(trans_vals, parents=[self], op='transpose', auto_grad=True)
        output.op_params = {'axes': axes}
        return output

    def flatten(self):
        flatted_vals = flatten(self.data)
        output = Tensor(flatted_vals, parents=[self], op='flatten', auto_grad=True)
        return output


    @staticmethod
    def concatenate(tensors, axis):
        tensors_data = [tensor.data for tensor in tensors]
        concate_vals = np.concatenate(tensors_data, axis)
    
        output = Tensor(concate_vals, parents=tensors, op='concatenate', auto_grad=True)
        output.op_params = {'axis': axis}
        return output

    # im2col function
    def tensor_to_matrix(self, ksize, stride=1, pad=0):
        matrix = tensor_to_matrix(self.data, ksize, stride, pad)
        op_params = {'ksize': ksize, 
                     'stride': stride, 
                     'pad': pad}
        output = Tensor(matrix, parents=[self], op='tensor_to_matrix', auto_grad=True)
        output.op_params = op_params
        return output

    def select_array_indice(self, axis, i):
        selected_array = select_array_indice(self.data, axis, i)
        output = Tensor(selected_array, parents=[self], op='select_array_indice', auto_grad=True)
        output.op_params = {'axis': axis, 'i': i}
        return output


    def relu(self):
        relu_vals = relu(self.data)
        output = Tensor(relu_vals, parents=[self], op='relu', auto_grad=True)
        return output

    def sigmoid(self):
        sigmoid_vals = sigmoid(self.data)
        output = Tensor(sigmoid_vals, parents=[self], op='sigmoid', auto_grad=True)
        return output

    def tanh(self):
        tanh_vals = tanh(self.data)
        output = Tensor(tanh_vals, parents=[self], op='tanh', auto_grad=True)
        return output

    def activation(self, type):
        if type == 'relu':
            return self.relu()
        elif type == 'sigmoid':
            return self.sigmoid()
        elif type == 'tanh':
            return self.tanh()

    def softmax(self):
        softmax_vals = softmax(self.data)
        output = Tensor(softmax_vals, parents=[self], op='softmax', auto_grad=True)
        return output

    def cross_entropy(self, target):
        logits = softmax(self.data)
        labels = target.data
        delta = 1e-7
        batch_size = self.data.shape[0]
        cross_entropy = -np.sum(labels * np.log(logits + delta))/batch_size
        output = Tensor(cross_entropy, parents=[self, target], op='cross_entropy', auto_grad=True)
        output.op_params = {'logits': logits}
        return output

    def dropout(self, rate):
        mask = get_dropout_mask(self.data, rate)
        if self.auto_grad:
            output = Tensor(mask, parents=[self], op='dropout', auto_grad=True)
            output.op_params = {'mask': mask}
            return output

    def clean_dependencies(self):
        """clean node dependencies
        """
        self.children = {}
        if self.parents is not None:
            for parent in self.parents:
                parent.clean_dependencies()
    
    def create_dependencies(self):
        
        if self.parents is not None:
            for parent in self.parents:
                if self.id not in parent.children:
                    parent.children[self.id] = 0	
                parent.children[self.id] += 1
                parent.create_dependencies()
    
    def refresh_dependencies(self):
        self.clean_dependencies()
        self.create_dependencies()

