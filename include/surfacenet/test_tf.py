

import sys
import os

BASE_DIR = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(os.path.join(BASE_DIR, './as_rigid_as_possible'))


import utils.utils_basic as ub
import utils.utils_fun as uf
import models_tf as models

from importlib import reload
#import importlib as imp
reload(ub)
reload(uf)
reload(models)


model = models.MlpModel()


b = 32
n = 300
f = 600

import tensorflow as tf
#V = tf.Variable(tf.zeros([b, n, 3]), dtype=tf.float32)
#F = tf.Variable(tf.zeros([b, f, 3]), dtype=tf.int32)

L = tf.Variable(tf.zeros([b, n, 3]), dtype=tf.float32)
input = tf.Variable(tf.zeros([b, n, 6]), dtype=tf.float32)
mask = tf.Variable(tf.zeros([b, n, 1]), dtype=tf.float32)


output = model.forward(L, mask, input)