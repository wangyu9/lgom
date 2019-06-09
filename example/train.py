is_in_interactive_mode = '__file__' not in globals()
# this simple hack works for me.

#def is_in_interactive_mode():
#    # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
    #import __main__ as main
    #return hasattr(main, '__file__')
#    return not bool(getattr(sys, 'ps1', sys.flags.interactive))

import argparse
import os
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(os.path.join(BASE_DIR, '../include'))
# TODO: recursively append the repo
sys.path.append(os.path.join(BASE_DIR, '../include/pointnet'))
sys.path.append(os.path.join(BASE_DIR, '../include/pointnet/models'))
sys.path.append(os.path.join(BASE_DIR, '../include/pointnet/util'))
sys.path.append(os.path.join(BASE_DIR, '../include/surfacenet'))
sys.path.append(os.path.join(BASE_DIR, '../include/surfacenet/utils'))


import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

##################################
# Load Library and Parse Options #
##################################
import sys

print(sys.version)
print(tf.__version__)

if is_in_interactive_mode:
    parser = argparse.ArgumentParser(prog='PROG')
else:
    parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--cpu', type=bool, default=False, help='use CPU [default: False]')
parser.add_argument('--soft_placement', type=bool, default=True, help='Use soft device placement [default: True]')
parser.add_argument('--model', default='model_shrec_cls_101', help='Model name [default: model_shrec_cls_101]')
parser.add_argument('--log_dir', default='', help='Log dir [default: log/$model/$version]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--dataset', default=None, help='Dataset [default: Not Given]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoches to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch Size for training [default: 50]')
parser.add_argument('--task', default='cls', help='Task: cls, att or seg [default: cls]')

parser.add_argument('--snapshot', default='', help='model snapshot file path [example: log/model.ckpt]')
parser.add_argument('--save_period', type=int, default=10, help='frequency to save model')


#parser.add_argument('--tensor_n', type=int, default=1200, help='Tensor n Size for training [default: 1200]')
#parser.add_argument('--tensor_f', type=int, default=2400, help='Tensor n Size for training [default: 2400]')
#parser.add_argument('--tensor_a', type=int, default=30, help='Tensor n Size for training [default: 30]')
parser.add_argument('--max_adj', type=int, default=None, help='Tensor n Size for training [default: None (use the default size in the data)]')
parser.add_argument('--model_arg', default='', help='Argument to pass to the model [default: '']')
parser.add_argument('--profiler', type=bool, default=False, help='Whether or not to run profiler [default: False]')

parser.add_argument('--mute', type=bool, default=False, help='Mute the display [default: False]')

parser.add_argument('--analysis', type=bool, default=False, help='')

#parser.add_argument('--model', default='model_mesh_mnist_102', help='Model name [default: model_mesh_mnist_102]')
#parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
#parser.add_argument('--dataset', default='mnist/mnist-sub', help='Dataset [default: mnist-sub]')
#parser.add_argument('--max_epoch', type=int, default=251, help='Epoches to run [default: 251]')


# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')


if is_in_interactive_mode:
    # a hack to make it work on interactive IPython
    argv = '--gpu=0'
    if 1:
        # argv = '--gpu=0 --dataset=mnist/mnist-sub --model=model_mesh_mnist_sub_103'
        #argv = '--gpu=0 --dataset=shrec15/shrec-cls-v0.h5 --model=model_shrec_cls_101'
        #argv = '--gpu=0 --dataset=haggai/haggai-seg-v0.h5 --model=model_haggai_seg_101 --batch_size=2 --tensor_f=65536 --tensor_n=32768'
        # argv = '--gpu=0 --dataset=haggai/haggai-seg-v0.h5 --model=model_haggai_seg_pointnet --batch_size=2 --tensor_f=65536 --tensor_n=32768 --max_epoch=40' # '--snapshot=log/trained_models/epoch_1.ckpt'
        #argv = '--gpu=0 --dataset=haggai/haggai-seg-v0.h5 --model=model_haggai_seg_101 --batch_size=2 --tensor_f=65536 --tensor_n=32768 --max_epoch=40'
        #argv = '--gpu=0 --dataset=shrec15/shrec-cls-v0.h5 --model=model_shrec_cls_101'
        # argv = '--gpu=0 --dataset=gp-test.mat --model=model_shrec_cls_101 --tensor_f=24000 --tensor_n=12000 --tensor_a=32'
        argv = '--gpu=0 --dataset=haggai/haggai-seg-v0.h5 --model=model_learn_hks_time --batch_size=2 --tensor_f=65536 --tensor_n=32768 --max_epoch=40 --learning_rate=1e-3' # --snapshot=log/model_learn_hks_time/34/trained_models/epoch_20.ckpt'
        #argv = '--gpu=0 --dataset=haggai/haggai-seg-v0.h5 --model=model_haggai_seg_pointnet3 --batch_size=2 --tensor_f=65536 --tensor_n=32768 --max_epoch=40'  # '--snapshot=log/trained_models/epoch_1.ckpt'
        argv = '--gpu=0 --dataset=modelnet10-v1.h5 --model=model_modelnet10_cls_pn_101 --batch_size=10 --tensor_f=4096 --tensor_n=2048 --tensor_a=90 --max_epoch=1 '
        argv = '--gpu=0 --dataset=modelnet40-v2.h5 --model=model_modelnet40_cls_pn_101 --batch_size=10 --tensor_f=8704 --tensor_n=4352 --tensor_a=30 --max_epoch=0 '
        argv = '--gpu=0 --dataset=modelnet40-debug-v2   --model=model_modelnet40_cls_pn_101 --batch_size=10 --tensor_f=2048 --tensor_n=3072 --tensor_a=30 --max_epoch=40 '
        # argv = '--gpu=0 --dataset=modelnet40-mini-v2.h5   --model=model_modelnet40_cls_102 --batch_size=5 --tensor_f=2048 --tensor_n=3072 --tensor_a=30 --max_epoch=40 '
        argv = '--gpu=0 --model=model_modelnet40_cls_lap_101 --batch_size=32 --max_epoch=10'
        #
        argv = '--gpu=0 --model=model_modelnet40_cls_pn_101 --batch_size=32 --max_epoch=10'
        #
        argv = '--gpu=0 --model=model_modelnet40_cls_lap_103 --batch_size=6 --max_epoch=200 --max_adj=8'

        argv = '--gpu=0 --model=model_profiler --profiler=True --batch_size=16'

        argv = '--gpu=0 --model=model_modelnet40_cls_pn_101 --batch_size=12 --max_epoch=200' # batch_size=50
        argv = '--gpu=0 --model=model_modelnet40_cls_lap_118 --batch_size=12 --max_adj=8'
        argv = '--gpu=0 --model=model_shrec_cls_lap_111 --batch_size=12 --max_adj=16'

        argv = '--gpu=0 --model=model_modelnet40_cls_lap_111 --batch_size=12 --max_adj=8'

        #argv = '--gpu=0 --model=model_shrec_cls_lap_207 --batch_size=12 --max_adj=16'

        argv = '--gpu=0 --model=model_shrec_cls_lap_152-4shallow-m --batch_size=12 --max_adj=16 --max_epoch=600'


        argv = '--gpu=0 --model=model_modelnet40_cls_lap_221-1 --batch_size=12 --max_adj=16 --max_epoch=4000'
        argv = '--gpu=0 --dataset=modelnet40-v3   --model=model_modelnet40_cls_pn_101 --batch_size=50 --max_epoch=10 '


        argv = '--gpu=0 --model=model_shrec_cls_lap_221 --batch_size=12 --max_adj=16 --max_epoch=4000'  # --snapshot=log/model_shrec_cls_lap_152/1/epoch_250.ckpt'


        argv = '--gpu=0 --model=model_shrec_attr_241r --batch_size=16 --max_adj=16 --max_epoch=4000'
        argv = '--gpu=0 --model=model_surfnet_arap_nop_101_relu --batch_size=12 --max_epoch=4000 --dataset=sn_arap_data_sample '  #
        argv = '--gpu=0 --model=model_shrec_cls_wd_245rli --batch_size=8 --learning_rate=0.1 --max_adj=16 --max_epoch=4000'


        argv = '--gpu=0 --model=model_shrec_cls_lap_112b --batch_size=2 --max_epoch=2'
        argv = '--gpu=0 --model=model_valid_surface_net --batch_size=2 --max_adj=40 --dataset=sn_arap_data_sample'
        argv = '--gpu=0 --model=model_surfnet_arap_dir_101 --batch_size=2 --max_epoch=4000 --dataset=sn_arap_data_sample '  #
        argv = '--gpu=0 --model=model_mnist_cls_241zld-48-in --batch_size=32 --max_adj=16 --max_epoch=4000 --save_period=1 --dataset=mnist-mesh-comp' # --dataset=mnist-mesh'
        argv = '--gpu=0 --model=model_shrec_cls_241 --batch_size=2 --max_epoch=2'
        # argv = '--gpu=0 --model=model_shrec_setup2_245rle --batch_size=2 --max_epoch=2'


        argv = '--gpu=0 --model=model_shrec_setup3_aug_wd_245rli_sip --batch_size=2 --max_epoch=2'
        argv = '--model=model_shrec_cls_wd_245rlep_analysis --batch_size=2 --snapshot='
        argv = '--model=model_surfnet_arap_dir_lr_101 --batch_size=10  --analysis=True --snapshot=log/gdpm/model_surfnet_arap_dir_lr_101/22/old3/epoch_191.ckpt --dataset=sn_arap_data_sample'
        argv = '--gpu=0 --model=model_basic_geometry --batch_size=2 --max_adj=40 '  # this is a static model


    print('argv=',argv)
    FLAGS = parser.parse_args(argv.split())

else:
    FLAGS = parser.parse_args()



MODEL = FLAGS.model
MODEL_FILE = os.path.join(BASE_DIR, MODEL+'.py')

MAX_EPOCH = FLAGS.max_epoch
SNAPSHOT = FLAGS.snapshot
LOG_DIR = FLAGS.log_dir
SAVE_PERIOD = FLAGS.save_period

LEARNING_RATE = FLAGS.learning_rate

if len(LOG_DIR)==0:
    check_dir('log')
    check_dir('log/' + MODEL)
    for i in range(60000):
        dir = 'log/' + MODEL + '/%d'%(i)
        if not os.path.exists(dir):
            LOG_DIR = dir
            break

check_dir(LOG_DIR)
# check_dir(LOG_DIR+'/include/')
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp -r ../include/ %s/' % (LOG_DIR)) # bkp of my main package
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

if True:
    # https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file

    import logging
    # from tensorflow import logging
    # logfilename = 'tensorflow.log'
    logfilename = os.path.join(LOG_DIR, 'tensorflow.log')

    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


DATASET = FLAGS.dataset
BATCH_SIZE = FLAGS.batch_size
TASK = FLAGS.task
MODEL_ARG = FLAGS.model_arg

MODEL_STORAGE_PATH = os.path.join(LOG_DIR, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

def log_print(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if not FLAGS.mute:
        print(out_str)


GPU_INDEX = FLAGS.gpu

if FLAGS.cpu:
    GPU_INDEX = None

SOFT_PLACEMENT = FLAGS.soft_placement


if GPU_INDEX is None:
    log_print('Use CPU')
else:
    log_print('Use GPU %d'%GPU_INDEX)



if is_in_interactive_mode:
    log_print('Interactive Python Mode')
    log_print('argv=%s'%argv)
else:
    log_print('Non-Interactive Python Mode')
    print('sys.argv=%s'%sys.argv)
    log_print('sys.argv=%s'%sys.argv)

enable_profiler = FLAGS.profiler

##############################################

#import tensorflow as tf
import numpy as np
import scipy.io as sio
import imp
import surf_util as su
import surf_struct as ss

def reload(module):
    if is_in_interactive_mode:
        imp.reload(module)

# sess = tf.InteractiveSession()



# scp "/Users/yuwan/Dropbox (MIT)/WorkSpace/research/dg/shrec15data.mat"
# yuwan@ilcompm1:~/WorkSpace/projects/subnet/data/shrec15/shrec15data.mat


#mat = sio.loadmat('../data/'+DATASET) # +'.mat'

def get_default_path():
    import socket
    hostname = socket.gethostname()

    if hostname == 'DESKTOP-H29QM4F':
        return 'D:/WorkSpace/vmware/shared/ubuntu/WorkSpace/dg/util/include/'
    elif hostname == 'pekseg-Precision-Tower-7910':
        return '/media/pekseg/sdb-2T-disk/home/yu/WorkSpace/dg/dataset/'
    else:
        #if os.path.exists('../data/'+DATASET):
        return '../data/'
        #else:
        #    print('Error: cannot find dataset path.')


# moved to part of model DATASET is no longer used.
# dataset = get_default_path() + DATASET

#n = FLAGS.tensor_n
#f = FLAGS.tensor_f
#a = FLAGS.tensor_a


# GPU setup was here

#################################
#   Load the Models             #
#################################

import surf_net as sn
import surf_model as sm

import surf_basic
import surf_op
import surf_dg

model = importlib.import_module(MODEL) # import network module

if DATASET is None:
    dataset =  get_default_path() + model.get_dataset()
else:
    print('Command arg overwrites model parameter %s with %s'%('dataset',DATASET))
    dataset = get_default_path() + DATASET

if True:
    reload(sm)
    reload(model)


def get_model():
    return model.get_model()


def get_mesh_level():
    if hasattr(model, 'get_mesh_level'):
        return model.get_mesh_level()
    else:
        return None


def get_mesh_struct():
    if hasattr(model, 'get_mesh_struct'):
        return model.get_mesh_struct()
    else:
        return ss.MeshStruct

#################################
#   Load the Mesh Struct        #
#################################

if False:
    reload(su)

# The mesh data struct was here.

#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
#bytes_in_use = BytesInUse()

def print_memory_usage():
    print('Memory not implemented.')
    # log_print('Memory in use: %d'%(sess.run(bytes_in_use)))
    # log_print()

#################################
#   Static Model Running        #
#################################

reload(model)

if hasattr(model, 'get_static_model'):

    mesh_train = get_mesh_struct()(dataset, 'train', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)
    mesh_test  = get_mesh_struct()(dataset, 'test', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)

    sess = None

    if enable_profiler:
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()
        pctx = tf.contrib.tfprof.ProfileContext(LOG_DIR,
                                                trace_steps=[],
                                                dump_steps=[])
        pctx_handle = pctx.__enter__()

    if sess is None:
        # Disable GPU memory pre-allocation using TF session configuration:

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = SOFT_PLACEMENT
        sess = tf.Session(config=config)
        sess_handle = sess.__enter__()

    if True:

        tf.global_variables_initializer().run()
        mesh_train.init_tf(BATCH_SIZE)
        metadata = ss.MetaData()
        metadata.init_tf(BATCH_SIZE)
        out_train = model.get_static_model()(mesh_train, metadata, MODEL_ARG)  # it coud be e.g. logits_train
        print_memory_usage()

        if enable_profiler:
            pctx.trace_next_step()
            # Dump the profile to '/tmp/train_dir' after the step.
            pctx.dump_next_step()
            #
            feed_train = mesh_train.make_batch(batch_index=0, augmentor=None)

            _ = sess.run(out_train.prof, feed_dict=feed_train)
            # profile result
            pctx.profiler.profile_operations(options=opts)

        reload(model)
        static = model.test_static_model(out_train=out_train, mesh_train=mesh_train)

        if False:
            if enable_profiler:
                pctx_handle.__exit__(None, None, None)
            sess_handle.__exit__(None, None, None)


if FLAGS.analysis or hasattr(model, 'get_analysis'):

    graph = tf.Graph()

    graph_handle = graph.as_default()
    graph_handle.__enter__()

    if GPU_INDEX is None:
        device = tf.device('/cpu:0')
    else:
        # Important: this device has to come after graph entering, or it will not take into effect.
        device = tf.device('/gpu:' + str(GPU_INDEX))
    if not is_in_interactive_mode:
        device_handle = device.__enter__()  # this breaks my static testing.

    mesh_train = get_mesh_struct()(dataset, 'train', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)
    mesh_test  = get_mesh_struct()(dataset, 'test', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)


    metadata = ss.MetaData()
    metadata.init_tf(BATCH_SIZE, base_lr=LEARNING_RATE)

    mesh_train.init_tf(BATCH_SIZE)
    mesh_test.init_tf(BATCH_SIZE, external_minst_tf_data=mesh_train)

    N_train = mesh_train.get_num_data()
    N_test  = mesh_test.get_num_data()

    out_train = get_model()(mesh_train, metadata, MODEL_ARG) # it coud be e.g. logits_train
    # out_test  = get_model()(mesh_test, metadata, MODEL_ARG)

    '''
    losses_train = get_loss()(mesh_train, out_train)
    losses_train = sn.collect_and_convert_losses(losses_train)
    assert 'main' in losses_train
    assert 'total' in losses_train

    # loss_test  = get_loss()(mesh_test,  out_test)

    train_op = get_train_op()(N_train, N_test, BATCH_SIZE, out_train, log_print)

    augmentor = get_augmentor()()
    test_noiser = get_test_noiser()()

    #################################
    #   Do the Training             #
    #################################

    optimizer = get_optimizer()(losses_train, metadata)
    '''

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = SOFT_PLACEMENT


    sess = tf.Session(config=config)
    sess_handle = sess.__enter__()

    index = 0

    #with tf.Session(config=config) as sess:
    if True:

        if len(SNAPSHOT)>0:
            saver.restore(sess, SNAPSHOT)
            log_print("Model restored.")
        else:
            # tf.global_variables_initializer().run()
            init = tf.global_variables_initializer()
            sess.run(init)


#################################
#   Start Model Training        #
#################################
if not hasattr(model, 'get_model') or FLAGS.analysis:
    log_print('Warning: no trainable model give!')
else:

    def get_loss():
        return model.get_loss()


    def get_train_op():
        return model.get_train_op()


    def get_augmentor():
        if hasattr(model, 'get_augmentor'):
            return model.get_augmentor()
        else:
            return ss.Augmentor

    def get_test_noiser():
        if hasattr(model, 'get_test_noiser'):
            return model.get_test_noiser()
        else:
            return ss.TestNoiser


    def get_optimizer():
        if hasattr(model, 'get_optimizer'):
            return model.get_optimizer()
        else:
            return ss.Optimizer

    if enable_profiler:
        # from tensorflow.contrib.tfprof import model_analyzer
        # profiler =  model_analyzer.Profiler(graph)

        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md#demo
        # Create options to profile the time and memory information.
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()
        # Create a profiling context, set constructor argument `trace_steps`,
        # `dump_steps` to empty for explicit control.
        pctx = tf.contrib.tfprof.ProfileContext(LOG_DIR,
                                                trace_steps=[],
                                                dump_steps=[])
        pctx_handle = pctx.__enter__()

    else:

        graph = tf.Graph()

        graph_handle = graph.as_default()
        graph_handle.__enter__()
        #with graph.as_default():

    if GPU_INDEX is None:
        device = tf.device('/cpu:0')
    else:
        # Important: this device has to come after graph entering, or it will not take into effect.
        device = tf.device('/gpu:' + str(GPU_INDEX))
    if not is_in_interactive_mode:
        device_handle = device.__enter__()  # this breaks my static testing.

    mesh_train = get_mesh_struct()(dataset, 'train', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)
    mesh_test  = get_mesh_struct()(dataset, 'test', mesh_level=get_mesh_level(), max_adj=FLAGS.max_adj)

    if True:

        #################################
        #   TF data are set             #
        #################################

        metadata = ss.MetaData()
        metadata.init_tf(BATCH_SIZE, base_lr=LEARNING_RATE)

        mesh_train.init_tf(BATCH_SIZE)
        mesh_test.init_tf(BATCH_SIZE, external_minst_tf_data=mesh_train)

        N_train = mesh_train.get_num_data()
        N_test  = mesh_test.get_num_data()

        out_train = get_model()(mesh_train, metadata, MODEL_ARG) # it coud be e.g. logits_train
        # out_test  = get_model()(mesh_test, metadata, MODEL_ARG)

        losses_train = get_loss()(mesh_train, out_train)
        losses_train = sn.collect_and_convert_losses(losses_train)
        assert 'main' in losses_train
        assert 'total' in losses_train

        # loss_test  = get_loss()(mesh_test,  out_test)

        train_op = get_train_op()(N_train, N_test, BATCH_SIZE, out_train, log_print)

        augmentor = get_augmentor()()
        test_noiser = get_test_noiser()()

        #################################
        #   Do the Training             #
        #################################

        optimizer = get_optimizer()(losses_train, metadata)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = SOFT_PLACEMENT

        sess = tf.Session(config=config)
        sess_handle = sess.__enter__()

        index = 0

        #with tf.Session(config=config) as sess:
        if True:

            if len(SNAPSHOT)>0:
                saver.restore(sess, SNAPSHOT)
                log_print("Model restored.")
            else:
                # tf.global_variables_initializer().run()
                init = tf.global_variables_initializer()
                sess.run(init)

            print_memory_usage()

            # train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
            # test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

            # downed_sample_mesh0 = pooled.eval(feed_dict=feed_train)

            for epoch in range(MAX_EPOCH):

                log_print("Epoch: %d" % (epoch))
                # TODO: get this back!! train_op.pre_epoch(graph)

                if False and MODEL[:18]=='model_surfnet_arap':
                    for batch_idx in range(0, N_test // BATCH_SIZE):
                        feed_test = mesh_test.make_batch(batch_idx)  # no augmentor as testing.
                        feed_test.update(metadata.make_batch(batch_idx, is_training=False))

                        train_op.update_val(batch_idx, feed_test, losses_train)

                # Training
                if not ( len(SNAPSHOT) > 0 and epoch==0): # test it first if loaded from snapshot.

                    for batch_idx in range(0,N_train//BATCH_SIZE):

                        feed_train = mesh_train.make_batch(batch_idx, augmentor=augmentor)
                        feed_train.update(metadata.make_batch(batch_idx, is_training=True))

                        if enable_profiler and epoch % 10 == 0:
                            '''
                            # https://www.tensorflow.org/api_docs/python/tf/profiler/Profiler
                            run_meta = tf.RunMetadata()
                            sess.run([optimizer.train_step], feed_dict=feed_train,
                                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                     run_metadata=run_meta)
                            profiler.add_step(index, run_meta)

                            # Profile the parameters of your model.
                            profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
                                                                 .trainable_variables_parameter()))

                            # Or profile the timing of your model operations.
                            opts = option_builder.ProfileOptionBuilder.time_and_memory()
                            profiler.profile_operations(options=opts)

                            # Or you can generate a timeline:
                            opts = (option_builder.ProfileOptionBuilder(
                                option_builder.ProfileOptionBuilder.time_and_memory())
                                    .with_step(i)
                                    .with_timeline_output(filename).build())
                            profiler.profile_graph(options=opts)
                            '''

                            pctx.trace_next_step()
                            # Dump the profile to '/tmp/train_dir' after the step.
                            pctx.dump_next_step()

                            sess.run([optimizer.train_step], feed_dict=feed_train)

                            pctx.profiler.profile_operations(options=opts)

                        else:

                            # use the right train_step
                            sess.run([optimizer.train_step], feed_dict=feed_train)

                        train_op.update( batch_idx, feed_train, losses_train)

                    train_op.update_epoch( mesh_train)

                # Validation

                for batch_idx in range(0, N_test // BATCH_SIZE):

                    feed_test = mesh_test.make_batch(batch_idx, augmentor=test_noiser) # no augmentor as testing.
                    feed_test.update(metadata.make_batch(batch_idx, is_training=False))

                    train_op.update_val( batch_idx, feed_test, losses_train)

                train_op.update_epoch_val( mesh_test)

                if epoch >= 0 and epoch % SAVE_PERIOD == 0:
                    cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.ckpt'))
                    log_print( 'Successfully store the checkpoint model into ' + cp_filename)

                index = index + 1

            def show_mesh_batch(mesh_batch, label_batch, idx=0):

                TF = mesh_batch.TF[idx,:,:]

                b = np.where(np.min(TF,axis=1)>=0)[0]

                log_print(TF[b,:].shape)
                log_print('Label: %d'%label_batch.TY[idx])

                su.show_mesh( mesh_batch.TV[idx,:,:], TF[ b, : ],color=0.1*np.ones_like(TF[b,:]))

            if False:# is_in_interactive_mode:
                s = 0
                show_mesh_batch( mesh_train.mesh.make_batch(s)[0], mesh_train.label.make_batch(s)[0], idx=6)

            if False:
                sess_handle.__exit__(None, None, None)

        if enable_profiler:
            pctx_handle.__exit__(None, None, None)

    if False:
        graph_handle.__exit__(None, None, None)



    if False:
        sess_handle.__exit__(None,None,None)
        graph_handle.__exit__(None,None,None)

        reload(surf_basic)
        reload(surf_op)
        reload(surf_dg)

        reload(sn)
        reload(sm)
        reload(ss)

        reload(model)


    # Get all variable names in the graph
    # [n.name for n in tf.get_default_graph().as_graph_def().node]

    if False:
        f = open('tmp.txt','w')
        #f.write('hellow world')
        for n in tf.get_default_graph().as_graph_def().node:
            f.write('%s\n'%(n.name))
        f.close()

if not is_in_interactive_mode:
    device_handle.__exit__(None, None, None)

