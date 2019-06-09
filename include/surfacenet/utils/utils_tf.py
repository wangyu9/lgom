'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import os
import sys

sys.path.append(os.path.dirname(__file__))

# original not used: import scipy as sp
import numpy as np
# original not used: import utils.graph as graph


# moved into basic.py: cuda.sparse_bmm_func import SparseBMMFunc

# tf: import torch
# tf: import torch.nn as nn
# tf: import torch.nn.functional as F
# original not used: from torch.autograd import Variable

import utils.utils_fun as uf
import utils.utils_basic as ub
NNModule = ub.Module


# tensors is a list of size batch
# this yields a sparse tensor of size [batch, size0, size1]
def sparse_cat(tensors, size0, size1):
    values = []
    tensor_size = 0
    for i, tensor in enumerate(tensors):
        values.append(tensor._values())
        tensor_size += tensor._nnz()

    import torch
    indices = torch.LongTensor(3, tensor_size)
    index = 0
    for i, tensor in enumerate(tensors):
        indices[0, index:index+tensor._nnz()] = i
        indices[1:3, index:index+tensor._nnz()].copy_(tensor._indices())

        index += tensor._nnz()

    values = torch.cat(values, 0)

    size = torch.Size((len(tensors), size0, size1))
    return torch.sparse.FloatTensor(indices, values, size).coalesce()


# wangyu
def np_sparse_to_pt_sparse(tensor, size0, size1):
    import torch
    return torch.sparse.FloatTensor(tensor._indices(), tensor._values(), torch.Size((size0, size1))).coalesce()


def sparse_tensor_array(tensors, size0, size1):
    return [np_sparse_to_pt_sparse(tensor, size0, size1) for _, tensor in enumerate(tensors)]



def sp_sparse_to_pt_sparse(L):
    """
    Converts a scipy matrix into a pytorch one.
    """
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))

    import torch
    indices = torch.from_numpy(indices).long()
    L_data = torch.from_numpy(L.data)

    size = torch.Size(L.shape)

    indices = indices.transpose(1, 0)

    L = torch.sparse.FloatTensor(indices, L_data, size)
    return L



def to_dense_batched(x, batch_size):
    x = x.to_dense()
    x = x.unsqueeze(0)
    return x.repeat(batch_size, 1, 1)



class GraphConv1x1(NNModule):
    def __init__(self, num_inputs, num_outputs, batch_norm=None, batch_fun=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm

        if self.batch_norm == "pre":
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(num_inputs)
            if batch_fun is None:
                self.bn = uf.BatchNorm1d(num_inputs)
            else:
                self.bn = batch_fun(num_inputs)

        if self.batch_norm == "post":
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(num_outputs)
            if batch_fun is None:
                self.bn = uf.BatchNorm1d(num_outputs)
            else:
                self.bn = batch_fun(num_outputs)
        #import torch.nn as nn
        #self.fc = nn.Linear(num_inputs, num_outputs)
        self.fc = uf.Linear(num_inputs, num_outputs)

    def forward(self, x):

        batch_size, num_nodes, num_inputs = ub.shape(x)  # x.size()
        assert num_inputs == self.num_inputs

        x = ub.contiguous(x)
        x = ub.reshape(x, shape=[-1, self.num_inputs])

        if self.batch_norm == "pre":
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(num_inputs)
            x = self.bn.forward(x)

        x = self.fc.forward(x)

        if self.batch_norm == "post":
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(self.num_outputs)
            x = self.bn.forward(x)

        x = ub.reshape(x, shape=[batch_size, num_nodes, self.num_outputs])

        return x


class GraphBatchNorm(NNModule):
    def __init__(self, num_inputs, batch_fun=None):
        super(GraphBatchNorm, self).__init__()

        self.num_inputs = num_inputs

        if batch_fun is None:
            self.bn = uf.BatchNorm1d(num_inputs)
        else:
            self.bn = batch_fun(num_inputs)

    def forward(self, x):
        #import torch.nn as nn
        #self.bn = nn.BatchNorm1d(self.num_inputs)

        batch_size, num_nodes, num_inputs = ub.shape(x)
        x = ub.reshape(x, shape=[batch_size * num_nodes, num_inputs])
        # todo: add back self.bn.train()
        x = self.bn.forward(x)

        x = ub.reshape(x, shape=[batch_size, num_nodes, num_inputs])
        return x


def global_average(x, mask):
    mask = ub.expand_as(mask, x) # mask.expand_as(x)
    # I do not understance why, but I need to use (x * mask) rather than x*mask here.
    return ub.reduce_sum((x * mask), axis=1, keepdims=True) / ub.reduce_sum(mask, axis=1, keepdims=True)


# Original Code:
#def global_average(x, mask):
#    mask = mask.expand_as(x)
#    return (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)


class LapResNet2(NNModule):
    def __init__(self, num_outputs, batch_fun):
        super(LapResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)

    def forward(self, L_matvec, mask, inputs, act_fn=uf.elu):
        # def forward(self, L, mask, inputs):

        x = inputs
        x = act_fn(x)

        #import torch
        #xs = [x, SparseBMMFunc()(L, x)]
        #x = torch.cat(xs, 2)

        #x = ub.concat([x, ub.sparse_matmul(L, x)], axis=2)
        x = ub.concat([x, L_matvec(x)], axis=2)
        x = self.bn_fc0.forward(x)
        x = act_fn(x)

        #xs = [x, SparseBMMFunc()(L, x)]
        #x = torch.cat(xs, 2)

        x = ub.concat([x, L_matvec(x)], axis=2)
        # x = ub.concat([x, ub.sparse_matmul(L, x)], axis=2)
        x = self.bn_fc1.forward(x)

        return x + inputs


class DirResNet2(NNModule):
    def __init__(self, num_outputs, res_f=False, batch_fun=None):
        super(DirResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)
        self.res_f = res_f

    def forward(self, Di_matvec, DiA_matvec, v, f, act_fn=uf.elu):
        #def forward(self, Di, DiA, v, f):
        # batch_size, num_nodes, num_inputs = v.size()
        batch_size, num_nodes, num_inputs = ub.shape(v)
        #_, num_faces, _ = f.size()
        _, num_faces, _ = ub.shape(f)
        x_in, f_in = act_fn(v), act_fn(f)

        #import torch

        x = x_in
        # x = x.view(batch_size, num_nodes * 4, num_inputs // 4)
        x = ub.reshape(x, shape=[batch_size, num_nodes * 4, num_inputs // 4])
        # x = SparseBMMFunc()(Di, x)
        # x = ub.sparse_matmul(Di, x)
        x = Di_matvec(x)
        #x = x.view(batch_size, num_faces, num_inputs)
        x = ub.reshape(x, shape=[batch_size, num_faces, num_inputs])
        #x = torch.cat([f_in, x], 2)
        x = ub.concat([f_in, x], axis=2)

        x = self.bn_fc0.forward(x)
        f_out = x

        x = act_fn(x)
        # x = x.view(batch_size, num_faces * 4, num_inputs // 4)
        x = ub.reshape(x, shape=[batch_size, num_faces * 4, num_inputs // 4])
        # x = SparseBMMFunc()(DiA, x)
        # x = ub.sparse_matmul(DiA, x)
        x = DiA_matvec(x)
        # x = x.view(batch_size, num_nodes, num_inputs)
        x = ub.reshape(x, shape=[batch_size, num_nodes, num_inputs])
        # x = torch.cat([x_in, x], 2)
        x = ub.concat([x_in, x], axis=2)

        x = self.bn_fc1.forward(x)
        v_out = x

        return v + v_out, f_out


class AvgResNet2(NNModule):
    def __init__(self, num_outputs, batch_fun):
        super(AvgResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre", batch_fun=batch_fun)

    def forward(self, L, mask, inputs, act_fn=uf.elu):
        x = inputs
        x = act_fn(x)

        #import torch

        #xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        xs = [x, ub.contiguous(ub.expand_as(global_average(x, mask), x))]

        #x = torch.cat(xs, 2)
        x = ub.concat(xs, axis=2)

        x = self.bn_fc0.forward(x)

        x = act_fn(x)
        # xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        xs = [x, ub.contiguous(ub.expand_as(global_average(x, mask), x))]
        # x = torch.cat(xs, 2)
        x = ub.concat(xs, axis=2)
        x = self.bn_fc1.forward(x)

        return x + inputs


class MlpResNet2(NNModule):
    def __init__(self, num_outputs, batch_fun):
        super(MlpResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn0 = GraphBatchNorm(num_outputs, batch_fun=batch_fun)
        self.fc0 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None, batch_fun=batch_fun)
        self.bn1 = GraphBatchNorm(num_outputs, batch_fun=batch_fun)
        self.fc1 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None, batch_fun=batch_fun)

    def forward(self, L, mask, inputs, act_fn=uf.elu):
        x = inputs
        x = self.bn0.forward(x)
        x = act_fn(x)
        x = self.fc0.forward(x)
        x = self.bn1.forward(x)
        x = act_fn(x)
        x = self.fc1.forward(x)
        return x + inputs
