import imp

from surf_basic import *
from surf_op import *
from surf_dg import *

#imp.reload(surf_basic)
#imp.reload(surf_op)
#imp.reload(surf_dg)

## Loss Functions

def matrix_norm_loss(x1,x2):

    diff = x1 - x2

    loss = tf.reduce_mean( tf.multiply(diff,diff) )

    return tf.sqrt(loss)


def matrix_norm_loss_upto_sign(x1,x2):

    diff0 = x1 - x2

    loss0 = tf.reduce_mean( tf.multiply(diff0,diff0) )
    
    diff1 = x1 + x2
    
    loss1 = tf.reduce_mean( tf.multiply(diff1,diff1) )
    
    loss = tf.minimum(loss0, loss1)

    return tf.sqrt(loss)


def collect_and_convert_losses(losses):
    def is_dict(d):
        # https://stackoverflow.com/questions/25231989/how-to-check-if-a-variable-is-a-dictionary-in-python
        import collections
        if isinstance(d, collections.Mapping):
            return True
        else:
            return False

    if is_dict(losses):
        assert 'main' in losses
        # cannot use  hasattr()
        if 'total' not in losses:
            loss_sum = tf.add_n([value for key, value in losses.items()])
            losses['total'] = loss_sum
    else:
        losses = {'total': losses, 'main': losses}

    return losses


class ModelOutput(object):
    def __init__(self, logits, *positional_parameters, ** keyword_parameters):

        self.logits = logits

        for kp in keyword_parameters:
            setattr(self, kp, keyword_parameters[kp])
        return
