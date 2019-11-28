from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..tensors import Tensor

def add_test():
    a = Tensor([1,2,3,4,5], auto_grad=True)
    b = Tensor([2,2,2,2,2], auto_grad=True)
    # c = Tensor([5,4,3,2,1], auto_grad=True)
    c = a + b
    print('Tensor add test: a + b = c')
    print('{} - {} = {}'.format(a, b, c))
    
    c.backward(Tensor(np.array([1,1,1,1,1])))
    print('gradient: {}'.format(c.grad))

    print(a.grad.data)
    print(b.grad.data)


def sub_test():
    a = Tensor([1,2,3,4,5], auto_grad=True)
    b = Tensor([2,2,2,2,2], auto_grad=True)
    c = a - b
    print('Test sub test: a + b = c')
    print('{} - {} = {}'.format(a, b, c))
    
    c.backward(Tensor(np.array([1,1,1,1,1])))
    print('gradient of c: {}'.format(c.grad.data))
    # print(c.grad.data)


def neg_test():
    a = Tensor([1,2,3,4,5], auto_grad=True)
    b = -a
    print('Tensor negative test: b = -a')
    print('{}'.format(b))
    b.backward(Tensor(np.array([1,1,1,1,1])))
    print('gradient of a: {}'.format(a.grad))
    # print(a.grad.data)



def multiply_test():
    a = Tensor([1,2,3,4,5], auto_grad=True)
    b = Tensor([2,2,2,2,2], auto_grad=True)
    c = a * b
    print('Tensor mutiplication: c = a * b')
    print('{} * {} = {}'.format(a, b, c))

    c.backward(Tensor(np.array([1,1,1,1,1])))
    print('gradient of c: {}'.format(c.grad.data))

    print('Gradient of a {} is equal to data of b {}: {}'.format(a.grad.data, b.data, np.all(a.grad.data == b.data)))


def matmul_test():
    a = Tensor(np.random.randint((2, 3)), auto_grad=True)
    b = Tensor(np.random.randint((2, 3)), auto_grad=True)
    c = a * b
    print(c)
    c.backward(Tensor(np.array([1, 1])))
    print(c.grad.data)
    

def power_test():
    a = np.array(range(12)).reshape(3,4)
    a = Tensor(a, auto_grad=True)
    print('a: \n {}'.format(a))

    b = a.__power__(num=2)
    print('b: \n {}'.format(b))

    print('power of a.data with p = 2 \n {}'.format(np.power(a.data, 2)))

    grad_data = np.ones_like(b.data)
    b.backward(Tensor(grad_data))
    print('Gradient of b \n {}'.format(b.grad.data))

def max_test():
    data = np.array(range(12)).reshape(3,4)
    a = Tensor(data, auto_grad=True)
    b = a.max()

    print(a.data)

    print(b.data)

    print(np.all(b.data == data.max(axis=1)))


def reshape_test():
    a = Tensor(np.random.randn(3, 4), auto_grad=True)
    b = a.reshape((4,3))

    print(a)
    print(b)
    b.backward(Tensor(np.ones(b.data.shape)))
    print(b.grad.data)
    print(np.all(a.grad.data == b.grad.data.reshape((3,4))))

def sum_test():
    a = Tensor(np.random.randn(3, 4), auto_grad=True)
    print(a)
    print(a.sum(0).data)
    print(a.sum(1).data)


def concatenate_test():
    a = Tensor(np.ones((3,1)), auto_grad=True)
    b = Tensor(np.ones((3,2))*2, auto_grad=True)
    c = Tensor(np.ones((3,3))*3, auto_grad=True)
    d = Tensor.concatenate([a, b, c], axis=1)
    print(d.data)

def transpose_test():
    a = Tensor(np.ones((3, 3, 1)), auto_grad=True)

    b = Tensor.transpose(a, (-1, 0, 1))

    print(a.data.shape)

    print(b.data.shape)

def sigmoid_test():
    a = Tensor(np.ones((3, 3)), auto_grad=True)
    b = Tensor.sigmoid(a)
    print(b.data)


def relu_test():
    a = Tensor(np.random.random((3, 3)), auto_grad=True)

    b = Tensor.relu(a)

    print(a.data)
    print(b.data)

if __name__ == '__main__':

    relu_test()
    
    sigmoid_test()
    
    transpose_test()
    
    concatenate_test()
    
    sum_test()
    
    reshape_test()
    
    max_test()
    
    power_test()
    
    matmul_test()
    
    multiply_test()
    
    neg_test()
    
    sub_test()
    
    add_test()

