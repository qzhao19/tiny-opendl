from ..tensors import Tensor
from ..losses


a = np.random.randint(0, 100, (3, 3))
b = np.random.randint(0, 100, (3, 3))
a_t = Tensor(np.random.random((3, 3)), auto_grad=True)
b_t = Tensor(np.random.random((3, 3)), auto_grad=True)

# np.argmax(a_t.data, 1)
print(a_t.data)
print(b_t.data)

SoftmaxCrossEntropy().backward(a_t, b_t)
