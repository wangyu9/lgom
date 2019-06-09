import tensorflow as tf

import tf_util
from transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


# dims: default =[64,128,1024]
# mlp_dims: [512,256,128]
# num_class is used by both seg and cls tasks.
def pointnet(point_cloud, num_class, dims, mlp_dims, is_training=True, bn_decay=None, seg_or_cls=True):

    print('pointnet')
    print(point_cloud)

    """ Classification PointNet, input is BxNx3, output BxNxnum_class """

    batch_size, num_point, dim_point = point_cloud.get_shape()
    assert(dim_point==3)

    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    # transform: [B,3,3].

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    # [B,N,3]
    # print(point_cloud_transformed)

    input_image = tf.expand_dims(point_cloud_transformed, -1)
    # [B,N,3,1]
    print('input_image')
    print(input_image)

    net = tf_util.conv2d(input_image, dims[0], [1, 3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    # TODO: something wrong here, it gives me [B,N,1,64]

    # [B,N,1,64]
    print('net')
    print(net)

    net = tf_util.conv2d(net, dims[0], [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    # [B,N,1,64]
    print(net)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=dims[0])
    # transform: [B,64,64].
    end_points['transform'] = transform

    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    # [B,N,64]

    print('Testing from here:')

    point_feat = tf.expand_dims(net_transformed, [2])
    print(point_feat)
    # [B,N,1,64]

    net = tf_util.conv2d(point_feat, dims[0], [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    # [B,N,1,64]
    print(net)

    net = tf_util.conv2d(net, dims[1], [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    # [B,N,1,128]
    print(net)

    net = tf_util.conv2d(net, dims[2], [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    # [B,N,1,1024]

    print(net)

    if seg_or_cls:

        global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                         padding='VALID', scope='maxpool')
        print(global_feat)
        # [B,1024]

        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        concat_feat = tf.concat(values=[point_feat, global_feat_expand], axis=3)
        print(concat_feat)
        # [B,N,64+1024,1]

        net = tf_util.conv2d(concat_feat, mlp_dims[0], [1, 1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        # [B,N,512,1]

        net = tf_util.conv2d(net, mlp_dims[1], [1, 1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        # [B,N,256,1]

        net = tf_util.conv2d(net, mlp_dims[2], [1, 1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        # [B,N,256,1]

        net = tf_util.conv2d(net, mlp_dims[2], [1, 1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay)
        # [B,N,128,1]

        net = tf_util.conv2d(net, num_class, [1, 1],
                             padding='VALID', stride=[1,1], activation_fn=None,
                             scope='conv10')
        # [B,N,num_class,1]

        net = tf.squeeze(net, [2]) # BxNxC
        # [B,N,num_class]

    else:

        # TODO: check the following code:

        # CLASS

        print('Dim outputs starts here:')

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')
        # [B,1,1,1024]
        print(net)

        net = tf.reshape(net, [batch_size, -1])
        # [B,1024]

        print(net)

        net = multiple_linear_perpecton(net, mlp_dims, is_training, bn_decay, num_class)

    return net, end_points


def multiple_linear_perpecton(net, mlp_dims, is_training, bn_decay, num_class):

    # [batch, 1024]

    net = tf_util.fully_connected(net, mlp_dims[0], bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # [batch, 512] # print(net)

    if True:
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
    else: # when I made is_learning True
        net = tf_util.dropout2(net, is_training=is_training, scope='dp1',
                          keep_prob=0.7)  # wangyu change original code, but no difference.
    # [batch, 512] # print(net)

    net = tf_util.fully_connected(net, mlp_dims[1], bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # [batch, 256] #  print(net)

    if True:
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    else: # when I made is_learning True
        net = tf_util.dropout2(net, is_training=is_training, scope='dp2', keep_prob=0.7)

    # [batch, 256] # print(net)

    net = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')
    # [batch, num_class] # print(net)

    return net

''' 
def get_model1( mesh0, mesh_attr, mesh1, mesh2, mesh3, mesh4, down01, down12, down23, down34, per_face_feature_fun=sn.per_face_feature):

    mesh0.K_succ = sn.succ(mesh0.K)
    mesh0.K_prev = sn.prev(mesh0.K)

    # mesh1.K_succ = sn.succ(mesh1.K)
    # mesh1.K_prev = sn.prev(mesh1.K)

    # mesh2.K_succ = sn.succ(mesh2.K)
    # mesh2.K_prev = sn.prev(mesh2.K)

    # mesh3.K_succ = sn.succ(mesh3.K)
    # mesh3.K_prev = sn.prev(mesh3.K)

    # mesh4.K_succ = sn.succ(mesh4.K)
    # mesh4.K_prev = sn.prev(mesh4.K)

    # mesh0.FM = sm.face_mask(mesh0.F)
    # mesh1.FM = sm.face_mask(mesh1.F)
    # mesh2.FM = sm.face_mask(mesh2.F)

    U = mesh0.V

    logits, mesh0.end_points = pointnet(U, is_training=True, bn_decay=None)

    return logits

def get_model2( mesh, mesh_attr, mesh_d, mesh_d2, mesh_d3, mesh_d4, ds01, ds12, ds23, ds34):
    return get_model1( mesh.tf_data, mesh_attr.tf_data, mesh_d.tf_data, mesh_d2.tf_data, mesh_d3.tf_data, mesh_d4.tf_data, ds01.tf_data, ds12.tf_data, ds23.tf_data, ds34.tf_data)

def get_model3( mesh_all, options=''):
    return get_model2( mesh_all.mesh, mesh_all.mesh_attr, mesh_all.mesh_d, mesh_all.mesh_d2, mesh_all.mesh_d3, mesh_all.mesh_d4, mesh_all.ds01, mesh_all.ds12, mesh_all.ds23, mesh_all.ds34)

def get_model():
    return get_model3
'''

import numpy as np


def regularization_loss(end_points):

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1])) - tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)

    return mat_diff_loss


def cls_loss(cls_pred, label):

    # print('cls_loss')
    # print(cls_pred) # [B,num_class]
    # print(label) # [B,1]
    batch = label.get_shape()[0].value
    assert(label.get_shape()[1].value==1)

    label = tf.reshape(label, shape=[batch])

    per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred, labels=label)
    return tf.reduce_mean(per_instance_label_loss)


def total_cls_loss(cls_pred, label, end_points):

    mat_diff_loss = regularization_loss(end_points)
    return cls_loss(cls_pred, label) + mat_diff_loss * 1e-3


def seg_loss(seg_pred, seg):
    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num

    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
    return tf.reduce_mean(per_instance_seg_loss)


def total_seg_loss(seg_pred, seg, end_points):

    mat_diff_loss = regularization_loss(end_points)
    return seg_loss(seg_pred, seg) + mat_diff_loss * 1e-3

import surf_struct as ss


class Augmentor(ss.Augmentor):
    def __init__(self):
        return
    #def start(self):
    #    return
    def apply(self, data):
        return data_augmentor(data)


def data_augmentor(batch_vertices):
    # batch_vertices: BxNx3

    assert(batch_vertices.shape[2]==3)

    rotated_data = rotate_point_cloud(batch_vertices)
    jittered_data = jitter_point_cloud(rotated_data)

    return jittered_data

##################################################
### From pointnet/provider.py
##################################################


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data
