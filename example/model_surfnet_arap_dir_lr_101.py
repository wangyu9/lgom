from model_surface_net import *
from task_surface_net_arap import *


def get_model():
    return ss.ModelWrapper(model=model_surface_net, level=MESH_LEVEL, # num_class=NUM_CLASS,
                           # optional ones:
                           name='model_surfnet_arap',
                           type='dir_lr',
                           keep='',
                           )


