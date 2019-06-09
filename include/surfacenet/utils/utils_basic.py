
USE_TF = True
import tensorflow as tf
import torch


# assert reload of / operator.

def shape(A):
    if USE_TF:
        return A.get_shape().as_list()
    else:
        return A.size()


def repeat(A, multiples):
    # multiples is a list
    assert len(multiples)
    if USE_TF:
        return tf.tile(A, multiples)
    else:
        # https://stackoverflow.com/questions/7527849/how-to-extract-parameters-from-a-list-and-pass-them-to-a-function-call
        return A.repeat(*multiples)


def contiguous(A):
    if USE_TF:
        return A
    else:
        return A.contiguous()


def reshape(A, shape):
    # shape is a list
    assert len(shape)
    if USE_TF:
        return tf.reshape(A, shape=shape)
    else:
        return A.view(*shape)


def concat(values, axis):
    assert len(values)
    if USE_TF:
        return tf.concat(values, axis=axis)
    else:
        return torch.cat(values, dim=axis)


def expand_as(source, target):
    # source to expand
    # target shape to have.
    if USE_TF:
        # https: // stackoverflow.com / questions / 51346819 / which - function - in -tensorflow - is -similar - to - expand -as- in -pytorch
        shape_s = shape(source)
        shape_t = shape(target)
        assert len(shape_s) == len(shape_t)
        multiples = [1 for i in range(len(shape_s))]
        for i in range(len(shape_s)):
            if shape_s[i]!=shape_t[i]:
                assert shape_s[i]==1
                multiples[i] = shape_t[i]
        return tf.tile(source, multiples=multiples)
        # Starting from tf 1.9
        #return tf.contrib.framework.broadcast_to()
    else:
        return source.expand_as(target)


def reduce_sum(A, axis, keepdims=False):
    if USE_TF:
        return tf.reduce_sum(A, axis=axis, keep_dims=keepdims)
    else:
        # does not pass return torch.sum(input=A, dim=axis, keep_dim=keepdims)
        return A.sum(axis, keepdim=keepdims)


def zeros(shape):
    if USE_TF:
        return tf.zeros(shape=shape)
    else:
        return

######
# Sparse Tensor

def batch_sparse_matmul(sp_as, bs):
    assert len(sp_as) == shape(bs)[0]
    if USE_TF:
        return concat([tf.sparse_tensor_dense_matmul(sp_as[i],bs[i, :, :]) for i in range(len(sp_as))], axis=0)
    else:
        return concat([torch.sparse.mm(sp_as[i],bs[i,:,:]) for i in range(len(sp_as))], axis=0)

#def
##            # This is batch-wise sparse dense multiply.
def batch_sparse_matmul2(sp_as, bs):
    from cuda.sparse_bmm_func import SparseBMMFunc
    return SparseBMMFunc()(sp_as, bs)



######


if USE_TF:
    class Module():
        def __init__(self):
            self._modules = {} # empty dict#{'': None}  # dict
            return

        def add_module(self, name, module):
            # setattr(self, name, module)
            self._modules[name] = module

        def parameters(self):
            #return sum([v.parameters() for k, v in self._modules.items()])
            r = []
            for k, v in self._modules.items():
                r = r + v.parameters()
            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                # https://stackoverflow.com/questions/5628084/test-if-a-class-is-inherited-from-another
                # https://stackoverflow.com/questions/30682791/python-asking-if-two-objects-are-the-same-class
                if issubclass(type(value),Module):
                    r = r + value.parameters()
            return r

else:
    import torch.nn as nn

    Module = nn.Module