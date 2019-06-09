'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

#import torch
#import torch.nn as nn
import sys

sys.path.append('..')

#import utils.graph as graph
#import utils.mesh as mesh


import utils.utils_tf as utils # import utils.utils_pt as utils # import utils.utils_tf as utils

import utils.utils_fun as uf

#
import utils.utils_basic as ub
NNModule = ub.Module


def is_list(lc):
    return type(lc) is list


class Model(NNModule):

    def __init__(self, batch_fun=None):
        super(Model, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None, batch_fun=batch_fun)

        for i in range(15):
            if i % 2 == 0:
                module = utils.LapResNet2(128, batch_fun=batch_fun)
            else:
                module = utils.AvgResNet2(128, batch_fun=batch_fun)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre", batch_fun=batch_fun)

    def forward(self, L, mask, inputs, same_op=True):

        _, num_nodes, _ = ub.shape(inputs)
        x = self.conv1.forward(inputs)

        for i in range(15):
            # x = self._modules['rn{}'.format(i)].forward(L[i] if is_list(L) else L, mask, x)
            x = self._modules['rn{}'.format(i)].forward(L if same_op else L[i], mask, x)

        x = uf.elu(x)
        x = self.conv2.forward(x)

        return x + ub.repeat(inputs[:, :, -3:], multiples=[1, 1, 40])


class AvgModel(NNModule):

    def __init__(self, batch_fun=None):
        super(AvgModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None, batch_fun=batch_fun)

        for i in range(15):
            module = utils.AvgResNet2(128, batch_fun=batch_fun)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre", batch_fun=batch_fun)

    def forward(self, L, mask, inputs, same_op=True):
        _, num_nodes, _ = ub.shape(inputs)
        x = self.conv1.forward(inputs)

        for i in range(15):
            # x = self._modules['rn{}'.format(i)].forward(L[i] if is_list(L) else L, mask, x)
            x = self._modules['rn{}'.format(i)].forward(L if same_op else L[i], mask, x)

        x = uf.elu(x)
        x = self.conv2.forward(x)

        return x + ub.repeat(inputs[:, :, -3:], multiples=[1, 1, 40])


class MlpModel(NNModule):

    def __init__(self, batch_fun=None):
        super(MlpModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None, batch_fun=batch_fun)

        for i in range(15):
            module = utils.MlpResNet2(128, batch_fun=batch_fun)
            self.add_module("rn{}".format(i), module)

        self.bn = utils.GraphBatchNorm(128, batch_fun=batch_fun)
        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm=None, batch_fun=batch_fun)

    def forward(self, L, mask, inputs, same_op=True):
        _, num_nodes, _ = ub.shape(inputs)
        x = self.conv1.forward(inputs)

        for i in range(15):
            # x = self._modules['rn{}'.format(i)].forward(L[i] if is_list(L) else L, mask, x)
            x = self._modules['rn{}'.format(i)].forward(L if same_op else L[i], mask, x)

        x = self.bn.forward(x)
        x = uf.elu(x)
        x = self.conv2.forward(x)

        return x + ub.repeat(inputs[:, :, -3:], multiples=[1, 1, 40])


# todo: handle torch change.
class DirModel(NNModule):

    def __init__(self, batch_fun=None):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None, batch_fun=batch_fun)

        for i in range(15):
            if i % 2 == 0:
                module = utils.DirResNet2(128, batch_fun=batch_fun)
            else:
                module = utils.AvgResNet2(128, batch_fun=batch_fun)

            self.add_module("rn{}".format(i), module)

        # this dropout is here but not used in their original code.
        #import torch.nn as nn
        #self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre", batch_fun=batch_fun)


    def forward(self, Di, DiA, mask, inputs, same_op=True):
        batch_size, num_nodes, _ = ub.shape(inputs)

        v = self.conv1.forward(inputs)

        num_faces = ub.shape(DiA)[2] // 4

        #import torch
        #from torch.autograd import Variable
        #f = Variable(torch.zeros(batch_size, num_faces, 128))
        f = ub.zeros(shape=[batch_size, num_faces, 128])

        if not ub.USE_TF:
            if v.is_cuda:
                f = f.cuda()

        for i in range(15):
            if i % 2 == 0:
                # v, f = self._modules['rn{}'.format(i)].forward(Di[i] if is_list(Di) else Di, DiA[i] if is_list(DiA) else DiA, v, f)
                v, f = self._modules['rn{}'.format(i)].forward(Di if same_op else Di[i], DiA if same_op else DiA[i], v, f)
            else:
                v = self._modules['rn{}'.format(i)].forward(None, mask, v)

        x = v
        x = uf.elu(x)
        x = self.conv2.forward(x)

        return x + ub.repeat(inputs[:, :, -3:], multiples=[1, 1, 40])
