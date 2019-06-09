###################################################
# Basic Definition Level
###################################################
import surf_struct as ss

MESH_LEVEL = 0

# TO implemented in its realization.
# def get_model():
#


# not implemented
# def get_augmentor():


def mesh_attr_loss(mesh_struct, out):
    mesh0 = mesh_struct.mesh.tf_data
    import surf_dg as sd
    #attr_truth = sd.per_face_normal(mesh0.V, mesh0.F)
    # todo: this function is not very efficient, use the new one!
    attr_truth = sd.per_vertex_normal(mesh0.V, mesh0.F, mesh0.J, mesh0.K)

    # todo: consider change naming of logits.
    return ss.attr_loss(None, attr_pred=out.logits, attr_truth=attr_truth)


def get_loss():
    return mesh_attr_loss


def get_train_op():
    return ss.AttrTrainOp


def get_mesh_level():
    return MESH_LEVEL


def get_dataset():
    return 'shrec-cls-v3'
###################################################
# Wrapper Level
###################################################

# from model_cls_lap import *