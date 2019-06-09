from model_cls_lap_240 import *
from task_mnist_cls import *


def get_model():
    return ss.ModelWrapper(model=model_cls_lap_24X, level=MESH_LEVEL, num_class=NUM_CLASS,
                           # optional ones:
                           name='model_mnist_cls_241',
                           pooling=sm.max_pooling,
                           keep='',
                           num_ops=[1,1,1],
                           setup_type='3',  # '3' stands for learnable operator.
                           learn_assembler=learn_assembler_1,
                           learn_slicer=learn_slicer_local_0,
                           op_cdims=[48, 48, 48],
                           Bf_rd=[1, 1/48, 1/48],
                           Bv_rd=[1/48, 1/48, 1/48],
                           dims=[48, 48, 48],
                           input_fun=input_fun_zs,
                           out_type='cls-dense',
                           )


# gcp-0: vim model_mnist_cls_241zld-48/3/log_train.txt
# line 30789: 98.5%


# gcp-a
# python3 train.py  --model=model_mnist_cls_241zld-48 --batch_size=50 --max_adj=16 --max_epoch=4000 --gpu=1
#

# a
# vim log/model_mnist_cls_241zld-48/13/log_train.txt
# acuracy: 98.8%
# line 50500

# gcp-a
# vim log/model_mnist_cls_241zld-48/7/log_train.txt
# 54677. : 99.01%

