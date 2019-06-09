###################################################
# Basic Definition Level
###################################################
import surf_struct as ss

NUM_CLASS = 50
MESH_LEVEL = 0

# TO implemented in its realization.
# def get_model():
#


# not implemented
# def get_augmentor():



def mesh_cls_loss_wd(mesh_struct, out):
    n_class = NUM_CLASS

    weights_decay = 1e-5
    import tensorflow as tf
    vars = tf.trainable_variables()
    pattern_end = 'feature_linear_trans/weights:0'
    w_list = [var for var in vars if var.name[-len(pattern_end):] == pattern_end]
    flt_weights_l2 = tf.add_n([tf.nn.l2_loss(w) for w in w_list])

    return {'main': ss.cls_loss(mesh_struct, out.logits, n_class), 'weights_decay': flt_weights_l2 * weights_decay}

    # old
    # return ss.cls_loss(mesh_struct, out.logits, n_class)


def mesh_cls_loss(mesh_struct, out):
    n_class = NUM_CLASS

    return ss.cls_loss(mesh_struct, out.logits, n_class)


def get_loss():
    return mesh_cls_loss


def get_train_op():
    return ss.SpecialClsTrainOp50


def get_mesh_level():
    return MESH_LEVEL


def get_dataset():
    return 'shrec-cls-v3'
###################################################
# Wrapper Level
###################################################

# from model_cls_lap import *


# sio.loadmat('D:\\WorkSpace\\vmware\\shared\\ubuntu\\WorkSpace\\dg\\util\\include\\shrec15\\y_label.mat')['y_label'][:,0]




#####################


