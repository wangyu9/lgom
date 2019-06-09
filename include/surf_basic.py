import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def weight_variable2(value,shape):
    initial = tf.constant(value,tf.float32,shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')


def batch_norm(x):
    return tf.layers.batch_normalization(x)


def get_tensor_shape(A):
    # tensorflow
    return A.get_shape().as_list()


def is_sparse(T):
    return type(T).__name__ is 'SparseTensor'


def batch_spread(A, multiples):
    [batch, one, _] = get_tensor_shape(A)
    assert one==1
    return tf.tile(A, multiples=[1,multiples,1])


def batch_inner_keep(A, B, Imatvec=None):
    # given two [batch,n,c] tensors A,B, compute their batch inner product keeping multiple channels.
    # Imatvec amounts to tensor: [batch,n,n], but it could be an abstract linear operator.
    if Imatvec==None: # this becomes the square L2 norm.
        return tf.expand_dims(tf.reduce_sum(tf.multiply(A, B), axis=1), axis=1)  # [batch,1,c]
    else:
        return tf.expand_dims(tf.reduce_sum(tf.multiply(A, Imatvec(B)), axis=1), axis=1) # [batch,1,c]


# seems expensive, avoid using.
def batch_inner_scatter(A, B, Imatvec=None):
    # given two tensors A,B, compute their batch outer.
    # [batch,n,c1], [batch,n,c2]
    if Imatvec==None: # this becomes the square L2 norm.
        MB = B
    else:
        MB = Imatvec(B)

    r = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.expand_dims(A, axis=-1), tf.expand_dims(MB, axis=-2)), axis=1), axis=1)  # [batch,1,c]
    [batch,one,_,_] = get_tensor_shape(r)
    assert one==1
    # [batch,1,c1 * c2]
    return tf.reshape(r, shape=[batch,1,-1])


def batch_matmul(A, B, Imatvec=None):
    # https://stackoverflow.com/a/43829731
    # given tensors A,B [batch,m,r] and [batch,r,n] , multiple them batch-wisely, yielding [batch,m,n].
    if Imatvec==None:
        return tf.matmul(A, B)
    else:
        # Imatvec should be [batch,r,r]
        return tf.matmul(A, Imatvec(B))


def batch_trans(A):
    # transpose a matrix batch-wisely
    return tf.transpose(A, perm=[0, 2, 1])


#def batch_diag(a):
#    # change a batch-wisely vector to a batch-wisely matrix.
#    # TODO


def batch_eye(batch, n):
    return tf.eye(n, batch_shape=[batch])


def batch_eye_like(B):
    shape = get_tensor_shape(B)
    return batch_eye(batch=shape[0], n=shape[1])


def batch_diagmul(a, B):
    # batch_matmul( batch_diag(a), B)
    # https://stackoverflow.com/questions/37904504/is-there-an-equivalent-bsxfun-in-tensorflow-as-there-in-matlab
    # Since Tensorflow does broadcasting.
    # return tf.multiply( a, B) #somehow this is still asking for:  but are 2 and 39 for 'Mul_58' (op: 'Mul') with input shapes: [2,39], [2,39,1].
    return tf.multiply( tf.expand_dims(a, axis=-1), B)


def batch_diaginv(a, B, epsilon=1e-10, epsilonB=0):
    #
    # old:
    # return batch_diagmul(tf.reciprocal(a+epsilon), B+epsilonB)
    return batch_diagmul(tf.where(tf.greater_equal(tf.abs(a),tf.abs(epsilon)), tf.reciprocal(a), tf.ones_like(a)*(1/epsilon)), B + epsilonB)


def batch_col_normalized(C, Imatvec=None, epsilon=1e-12):
    # [batch, n, c]
    return C / tf.sqrt(epsilon+batch_inner_keep(C, C, Imatvec=Imatvec))


def batch_sparse_matmul(As, x):
    batch_size = len(As)
    return tf.stack([tf.sparse_tensor_dense_matmul(As[bi], x[bi, :, :]) for bi in range(batch_size)], axis=0)

# Consider to merge the following code it make it more general.
#

def init_linear(in_dim, out_dim, use_bias):
    W = weight_variable([in_dim, out_dim])
    b = None
    if use_bias:
        b = bias_variable([1, out_dim])
    return W, b


def forward_linear(fun, W, b):

    [x, in_dim] = fun.get_shape().as_list()

    out_fun = tf.matmul(fun, W)

    if b is not None:
        be = b
        for i in range(len(get_tensor_shape(fun))-len(get_tensor_shape(b))):
            be = tf.expand_dims(be, axis=0)
        out_fun = out_fun + be

    # [x, out_dim]
    return out_fun


# This old implementation is extremely costly.
# Get rid of it.
def forward_linear_old(fun, W, b):

    [x, in_dim] = fun.get_shape().as_list()

    tmp = tf.tile(tf.expand_dims(W, 0), [x, 1, 1])

    # convert both to the size of [x, in_dim, out_dim]
    mul = tf.multiply(tf.expand_dims(fun, -1), tmp)

    out_fun = tf.reduce_sum(mul, axis=[1])

    if b is not None:
        out_fun = out_fun + b

    # [x, out_dim]
    return out_fun


def forward_batch_linear_old(fun, W, b):

    [batch, x, in_dim] = fun.get_shape().as_list()

    tmp = tf.tile(tf.expand_dims(tf.expand_dims(W, 0), 0), [batch, x, 1, 1])

    # convert both to the size of [batch, x, in_dim, out_dim]
    mul = tf.multiply(tf.expand_dims(fun, -1), tmp)

    out_fun = tf.reduce_sum(mul, axis=[2])

    if b is not None:
        out_fun = out_fun + b

    # [batch, x, out_dim]
    return out_fun



def init_batch_linear(in_dim, out_dim, use_bias):
    W = weight_variable([in_dim, out_dim])
    b = None
    if use_bias:
        b = bias_variable([1, 1, out_dim])
    return W, b


def batch_linear(fun, out_dim, use_bias=True):
    # [batch, x, in_dim]
    [batch, x, in_dim] = fun.get_shape().as_list()
    W, b = init_batch_linear(in_dim=in_dim, out_dim=out_dim, use_bias=use_bias)
    return forward_batch_linear(fun, W, b)


class Linear():
    def __init__(self, in_dim, out_dim, use_bias=True):
        # Usage is similar to:
        #import torch.nn as nn
        #self.fc = nn.Linear(in_dim, out_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W, self.b = init_linear(in_dim, out_dim, use_bias)

    # do not change the name, to be compatible to some torch module.
    def forward(self, x):

        return forward_linear(x, self.W, self.b)

    def parameters(self):

        return [self.W, self.b]