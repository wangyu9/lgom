from utils import utils_basic as ub

USE_TF = ub.USE_TF


def elu(x):
    if USE_TF:
        import tensorflow as tf
        return tf.nn.elu(x)
    else:
        import torch.nn.functional as F
        return F.elu(x)


def relu(x):
    if USE_TF:
        import tensorflow as tf
        return tf.nn.relu(x)
    else:
        import torch.nn.functional as F
        return F.relu(x)


if USE_TF:

    '''
    class BatchNorm1d():
        def __init__(self, size):
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(size)

            BatchNorm1d.num_instances = BatchNorm1d.num_instances + 1

            return

        def forward(self, x):
            #self.bn.train()
            #x = self.bn(x)
            import tf_util
            return tf_util.batch_norm_for_fc(x, True,
                                          bn_decay=None, scope='Class-BatchNorm1d-%d'%BatchNorm1d.num_instances)

        def parameters(self):
            return []

    BatchNorm1d.num_instances = 0
    '''


    import surf_basic as sb

    # copied from sb. consider merge later.
    # like https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
    class Linear(ub.Module):
        def __init__(self, in_dim, out_dim, use_bias=True):
            super(Linear, self).__init__() # note this!!!

            # Usage is similar to:
            # import torch.nn as nn
            # self.fc = nn.Linear(in_dim, out_dim)

            self.in_dim = in_dim
            self.out_dim = out_dim
            self.W, self.b = sb.init_linear(in_dim, out_dim, use_bias)

        # do not change the name, to be compatible to some torch module.
        def forward(self, x):
            return sb.forward_linear(x, self.W, self.b)

        def parameters(self):
            return [self.W, self.b]
            # this has to be a list


    class LinearBlock(ub.Module):
        def __init__(self, in_dim, out_dim, use_bias=True):
            super(LinearBlock, self).__init__() # note this!!!

            # Usage is similar to:
            # import torch.nn as nn
            # self.fc = nn.Linear(in_dim, out_dim)

            self.in_dim = in_dim
            self.out_dim = out_dim
            self.W, self.b = sb.init_linear(in_dim, out_dim, use_bias)

        # do not change the name, to be compatible to some torch module.
        def forward(self, x):
            return sb.forward_linear(x, self.W, self.b)

        def parameters(self):
            return [self.W, self.b]
            # this has to be a list

    assert hasattr(Linear, 'forward') # Important, this has forward.

    '''
    # this code cannot work since it breks some module functionalities.
    class BatchNorm1d():
        def __init__(self, size):
            import torch.nn as nn
            self.bn = nn.BatchNorm1d(size)
        def forward(self, x):
            self.bn.train()
            x = self.bn(x)
            return x

    class Linear():
        def __init__(self, in_dim, out_dim):
            import torch.nn as nn
            self.fc = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.fc(x)
    '''
else:
    import torch.nn as nn

    BatchNorm1d = nn.BatchNorm1d
    Linear = nn.Linear