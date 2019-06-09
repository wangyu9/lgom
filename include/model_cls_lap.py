
###################################################
from functools import partial as functional

def input_fun_positions(mesh, metadata=None):

    import surf_dg as sd
    if not hasattr(mesh, 'DA'):
        setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))

    return mesh.V


def get_vertices(mesh):
    if hasattr(mesh,'V'):
        return mesh.V
    else:
        return mesh.OV


def input_fun_rand20(mesh, metadata=None):

    import surf_basic as sb
    [batch, n, _] = sb.get_tensor_shape(get_vertices(mesh))

    if metadata is not None:
        w = tf.cond(metadata.is_training,
                lambda: 1.0,
                lambda: 0.0)
    else:
        w = 1.0

    U = tf.random_normal(shape=[batch, n, 20]) * w

    return U


def input_fun_rand(mesh, metadata=None):

    import surf_basic as sb
    [batch, n, _] = sb.get_tensor_shape(get_vertices(mesh))

    if metadata is not None:
        w = tf.cond(metadata.is_training,
                lambda: 1.0,
                lambda: 0.0)
    else:
        w = 1.0

    U = tf.random_normal(shape=[batch, n, 1]) * w

    return U


def learn_slicer_grad(mesh):
    import surf_dg as sd
    return sd.grad_slicer(mesh.V, mesh.F)


def input_fun_ones(mesh, metadata=None):

    import surf_dg as sd
    if not hasattr(mesh, 'DA'):
        setattr(mesh, 'DA', sd.per_face_doublearea(get_vertices(mesh), mesh.F))

    import surf_basic as sb

    [batch,n,_] = sb.get_tensor_shape(get_vertices(mesh))

    return tf.ones(shape=[batch,n,1])


def input_fun_zs(mesh, metadata=None):

    return tf.expand_dims(mesh.V[:,:,2], axis=-1)


def input_fun_plus(mesh, funs, attrs):

    # https: // stackoverflow.com / questions / 3061 / calling - a - function - of - a - module - by - using - its - name - a - string

    '''
    sm.set_per_face_wedge(mesh)
    sm.set_per_face_normal(mesh)
    sm.set_per_face_doublearea(mesh)
    sm.set_per_face_dir_edges(mesh)
    sm.set_per_face_edges_length(mesh)
    sm.set_per_face_angles(mesh)
    sm.set_per_face_cot(mesh)
    '''

    U = []

    if len(funs)>0:
        for fun in funs:
            set_fun = getattr(sm.set, fun)
            set_fun(mesh)

        if len(funs)>=2:
            face_aug_features = tf.stack([getattr(mesh, fun) for fun in funs], axis=-1)
            # [b,f,3,f_c]
        else:
            face_aug_features = getattr(mesh, funs[0])

        import surf_op as so
        VP = so.operator_face_aug_to_vertex(face_aug_features, mesh.J, mesh.K)
        # [b,n,k*f_c]
        U.append(VP)

    if len(attrs)>0:
        if len(attrs) >= 2:
            VA = tf.stack([getattr(mesh, att) for att in attrs], axis=-1)
        else:
            VA = getattr(mesh, attrs[0])
        U.append(VA)

    if len(U) >= 2:
        return tf.concat(U, axis=-1)
    else:
        return U[0]

###################################################
# Code Level
###################################################




import surf_model as sm

import tensorflow as tf
import surf_net as sn


def per_face_feature_xxx(V, F):

    def angle_from_edges(a, b, o):
        # the order of a,b does not matter.
        # o needs to be the edge opposite to the node.
        # assert a,b,o are of the same size.
        # the operation is applied element-wisely
        # the output have the same size as a,b,o.

        epsilon = 1e-10

        acos = tf.divide(tf.square(a) + tf.square(b) - tf.square(o), 2 * tf.multiply(a, b) + epsilon)

        return tf.acos(acos)

    def fmap( DE, W1, W2, W3, W4):

        def local_linear_trans(fun, out_dim, W, use_bias=True):
            # [batch, x, in_dim]
            [batch, x, in_dim] = fun.get_shape().as_list()

            # use weights from external
            # W = weight_variable([in_dim, out_dim])

            tmp = tf.tile(tf.expand_dims(tf.expand_dims(W, 0), 0), [batch, x, 1, 1])

            # convert both to the size of [batch, x, in_dim, out_dim]
            mul = tf.multiply(tf.expand_dims(fun, -1), tmp)

            out_fun = tf.reduce_sum(mul, axis=[2])

            if use_bias:
                b = sn.bias_variable([1, 1, out_dim])
                out_fun = out_fun + b

            # [batch, x, out_dim]
            return out_fun

        # [batch, f, 3]
        [batch, f, _] = DE.get_shape().as_list()

        F0 = DE # F0 = tf.reshape(DE, [batch, f, 3])

        F1 = local_linear_trans( F0, 12, W1, use_bias=True)

        F2 = tf.nn.relu( F1)

        F3 = local_linear_trans(F2, 10, W2, use_bias=True)

        F4 = tf.nn.relu(F3)

        F5 = local_linear_trans(F4, 10, W3, use_bias=True)

        F6 = tf.nn.relu(F5)

        F7 = local_linear_trans(F6, 1, W4, use_bias=True)

        F8 = tf.nn.relu(F7)
        # [batch, f, 1]

        return tf.reshape(F8, [batch, f])

    E = sn.per_face_edges_length(V, F)

    A = tf.stack([
        angle_from_edges( E[:,:,1], E[:,:,2], E[:,:,0]),
        angle_from_edges( E[:,:,2], E[:,:,0], E[:,:,1]),
        angle_from_edges( E[:,:,0], E[:,:,1], E[:,:,2])], axis=2)

    epsilon = 1e-10

    cotA = tf.reciprocal(tf.tan(A) + epsilon)

    # now learn a 9 --> 1 mapping: fmap

    # in this way, the weights are shared in all fmaps.
    W1 = sn.weight_variable([3, 12])
    # W1 = tf.expand_dims( tf.constant([0.0,0.0,1.0]), -1)
    W2 = sn.weight_variable([12, 10])
    W3 = sn.weight_variable([10, 10])
    W4 = sn.weight_variable([10, 1])

    C = tf.stack([
        fmap(tf.gather(A, [1, 2, 0], axis=2), W1, W2, W3, W4), # fmap(DE[:, :, [1, 2, 0], :]),
        fmap(tf.gather(A, [2, 0, 1], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [2, 0, 1], :]),
        fmap(tf.gather(A, [0, 1, 2], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [0, 1, 2], :])
    ], axis=2)

    # DE[:, :, 1, :], DE[:, :, 2, :], DE[:, :, 0, :]
    # DE[:, :, 2, :], DE[:, :, 0, :], DE[:, :, 1, :]
    # DE[:, :, 0, :], DE[:, :, 1, :], DE[:, :, 2, :]

    # [batch, f, 3]
    return C


import surf_basic as sb
gts = sb.get_tensor_shape


def dense_layer(fc0, dims, NUM_CLASS):
    mu = 0
    sigma = 0.1

    # dims = [50, 50]

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(gts(fc0)[1], dims[0]), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(dims[0]))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(dims[0], dims[1]), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(dims[1]))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(dims[1], NUM_CLASS), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(NUM_CLASS))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def check_K(mesh):
    if not hasattr(mesh, 'K_succ'):
        mesh.K_succ = sn.succ(mesh.K)
    if not hasattr(mesh, 'K_prev'):
        mesh.K_prev = sn.prev(mesh.K)
    if not hasattr(mesh, 'FM'):
        mesh.FM = sm.face_mask(mesh.F)


def dense_heat_kernel_signature(mesh):

    if not hasattr(mesh, 'K_succ'):
        mesh.K_succ = sn.succ(mesh.K)
    if not hasattr(mesh, 'K_prev'):
        mesh.K_prev = sn.prev(mesh.K)

    import surf_dg as sd
    # This is n.s.d.
    lap_op0 = sm.SparseLaplacianOperator(mesh, per_face_feature_fun=sd.per_face_cot)

    dlap = lap_op0.to_dense()


    if not hasattr(mesh, 'Mass'):
        mesh.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))


def block(U, mesh0, c1, is_training, bn_decay, per_face_feature_fun=sn.per_face_cot, mesh_face_fun=None, keep=''):

    if keep=='only_lap' or keep=='':

        #U01 = sm.laplacian_filter(U, mesh0)
        # the generalized_laplacian_filter consumes more memory somehow. Figure out why.
        # Z1 = sm.generalized_laplacian_filter_old(U, mesh0, per_face_feature_fun=per_face_feature_fun)
        Z1 = sm.generalized_laplacian_filter(U, mesh0, per_face_feature_fun=per_face_feature_fun)

        # [batch, n0, c0]
        with tf.variable_scope('lap') as sc:
            Z1 = sn.flt(Z1, c1, is_training, bn_decay, use_bias=True)

    #c1 = 10
    if keep=='only_linear':
        with tf.variable_scope('linear') as sc:
            Z2 = sn.flt(U, c1, is_training, bn_decay, use_bias=True)
        # [batch, n0, c1]

    if keep=='':
        with tf.variable_scope('linear') as sc:
            Z2 = sn.flt(U, c1, is_training, bn_decay, use_bias=False)
        # [batch, n0, c1]

    if keep=='only_lap':
        Z = Z1
    elif keep=='only_linear':
        Z = Z2
    else:
        assert(keep=='')
        Z = Z1 + Z2

    U03 = tf.nn.relu(Z)

    return U03


def conv_layer(U, mesh0, mesh1, mesh2, down01, down12): #, ):

    check_K(mesh0)
    check_K(mesh1)
    check_K(mesh2)

    Z = block(U, mesh0, 10) # c1=10

    Z = sm.avg_pooling(Z, mesh0, mesh1, down01)
    # [batch, n1, c1]

    Z = block(Z, mesh1, 4) # c2=10

    Z = sm.avg_pooling(Z, mesh1, mesh2, down12)
    # [batch, n1, c1]

    return Z


def max_pooling(net):

    b, n, c = gts(net)

    print('max_pooling')
    print(net)

    net = tf.expand_dims(net, 2)
    # make it [b, n, 1, c]

    import tf_util
    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [n, 1],
                             padding='VALID', scope='maxpool')
    # [b,1,1,c]
    print(net)

    net = tf.reshape(net, [b, -1])
    # [B,1024]

    print(net)

    return net


def per_face_feature_an(V, F):
    import surf_net as sn

    def angle_from_edges(a, b, o):
        # the order of a,b does not matter.
        # o needs to be the edge opposite to the node.
        # assert a,b,o are of the same size.
        # the operation is applied element-wisely
        # the output have the same size as a,b,o.

        epsilon = 1e-10

        acos = tf.divide(tf.square(a) + tf.square(b) - tf.square(o), 2 * tf.multiply(a, b) + epsilon)

        return tf.acos(acos)

    def fmap_old(DE):
        # [batch, f, 3]

        [batch, f, _] = DE.get_shape().as_list()

        F0 = DE  # F0 = tf.reshape(DE, [batch, f, 3])

        F1 = feature_linear_trans(F0, 10, use_bias=True)

        F2 = tf.nn.relu(F1)

        F3 = feature_linear_trans(F2, 10, use_bias=True)

        F4 = tf.nn.relu(F3)

        F5 = feature_linear_trans(F4, 10, use_bias=True)

        F6 = tf.nn.relu(F5)

        F7 = feature_linear_trans(F6, 1, use_bias=True)

        F8 = tf.nn.relu(F7)
        # [batch, f, 1]

        return tf.reshape(F8, [batch, f])

    E = sn.per_face_edges_length(V, F)

    A = tf.stack([
        angle_from_edges(E[:, :, 1], E[:, :, 2], E[:, :, 0]),
        angle_from_edges(E[:, :, 2], E[:, :, 0], E[:, :, 1]),
        angle_from_edges(E[:, :, 0], E[:, :, 1], E[:, :, 2])], axis=2)

    epsilon = 1e-10

    A = tf.reciprocal(tf.tan(A) + epsilon)

    # now learn a 9 --> 1 mapping: fmap

    # in this way, the weights are shared in all fmaps.
    W1 = sn.weight_variable([3, 10])
    # W1 = tf.expand_dims( tf.constant([0.0,0.0,1.0]), -1)
    W2 = sn.weight_variable([10, 10])
    W3 = sn.weight_variable([10, 10])
    W4 = sn.weight_variable([10, 1])

    C = tf.stack([
        local_fmap(tf.gather(A, [1, 2, 0], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [1, 2, 0], :]),
        local_fmap(tf.gather(A, [2, 0, 1], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [2, 0, 1], :]),
        local_fmap(tf.gather(A, [0, 1, 2], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [0, 1, 2], :])
    ], axis=2)

    # DE[:, :, 1, :], DE[:, :, 2, :], DE[:, :, 0, :]
    # DE[:, :, 2, :], DE[:, :, 0, :], DE[:, :, 1, :]
    # DE[:, :, 0, :], DE[:, :, 1, :], DE[:, :, 2, :]

    # [batch, f, 3]
    return C

# Since model cls lap 220

import surf_basic as sb

'''
def mop_block(U, mesh0, c1, is_training, bn_decay, per_face_feature_fun=sn.per_face_cot, keep=''):

    [_,_,c] = sb.get_tensor_shape(U)

    if not hasattr(mesh0, 'Mass'):
        mesh0.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh0.V, mesh0.F, mesh0.J))

    if keep=='only_lap' or keep=='':

        #U01 = sm.laplacian_filter(U, mesh0)
        # the generalized_laplacian_filter consumes more memory somehow. Figure out why.
        # Z1 = sm.generalized_laplacian_filter_old(U, mesh0, per_face_feature_fun=per_face_feature_fun)
        Z1 = tf.concat([sm.generalized_laplacian_filter(U[:,:,(4*i):(min(4*i+4,c))], mesh0, per_face_feature_fun=per_face_feature_fun) for i in range((c-1)//4+1)], axis=-1)
        sm.SparseFEMGeneral()

        Z1 = sb.batch_col_normalized(Z1, Imatvec=mesh0.Mass.matvec)

        # [batch, n0, c0]
        with tf.variable_scope('lap') as sc:
            #Z1 = sn.flt(Z1, c1, is_training, bn_decay, use_bias=True)

            Z1 = sn.feature_linear_trans_old(Z1, c1, use_bias=True)

    #c1 = 10
    if keep=='only_linear':
        with tf.variable_scope('linear') as sc:
            Z2 = sn.flt(U, c1, is_training, bn_decay, use_bias=True)
            #Z2 = sn.feature_linear_trans_old(U, c1, use_bias=True)
        # [batch, n0, c1]

    if keep=='':
        with tf.variable_scope('linear') as sc:
            Z2 = sn.flt(U, c1, is_training, bn_decay, use_bias=False)
            #Z2 = sn.feature_linear_trans_old(U, c1, use_bias=False)
        # [batch, n0, c1]

    if keep=='only_lap':
        Z = Z1
    elif keep=='only_linear':
        Z = Z2
    else:
        assert(keep=='')
        Z = Z1 + Z2

    #U03 = Z
    U03 = tf.nn.relu(Z)

    return U03
'''

# def recurrent_operator_layer

def apply_operator_baselinept(U, A, M_matvec, Minv_matvec, c1, is_training, bn_decay, column_normalize=True, keep=''):

    [_,_,c] = sb.get_tensor_shape(U)

    if keep=='only_lap' or keep=='':

        N1 = U#N1 = Minv_matvec(A.apply_to(U))
        # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

        #N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

        # [batch, n0, c0]
        with tf.variable_scope('lap') as sc:
            X1 = sn.flt(N1, c1, is_training, bn_decay, use_bias=True)
            #X1 = sn.feature_linear_trans_old(N1, c1, use_bias=True)

    #c1 = 10
    if keep=='only_linear':
        with tf.variable_scope('linear') as sc:
            X2 = sn.flt(U, c1, is_training, bn_decay, use_bias=True)
            #X2 = sn.feature_linear_trans_old(U, c1, use_bias=True)
        # [batch, n0, c1]

    if keep=='':
        with tf.variable_scope('linear') as sc:
            X2 = sn.flt(U, c1, is_training, bn_decay, use_bias=False)
            #X2 = sn.feature_linear_trans_old(U, c1, use_bias=False)
        # [batch, n0, c1]

    if keep=='only_lap':
        X = X1
    elif keep=='only_linear':
        X = X2
    else:
        assert(keep=='')
        X = X1 + X2

    # tf.nn.relu(X)
    return X


def apply_operator(U, A, M_matvec, Minv_matvec, c1, is_training, bn_decay, column_normalize=True, keep=''):

    [_,_,c] = sb.get_tensor_shape(U)

    if keep=='only_lap' or keep=='':

        N1 = Minv_matvec(A.apply_to(U))
        # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

        N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

        # [batch, n0, c0]
        with tf.variable_scope('lap') as sc:
            X1 = sn.flt(N1, c1, is_training, bn_decay, use_bias=True)
            #X1 = sn.feature_linear_trans_old(N1, c1, use_bias=True)

    #c1 = 10
    if keep=='only_linear':
        with tf.variable_scope('linear') as sc:
            X2 = sn.flt(U, c1, is_training, bn_decay, use_bias=True)
            #X2 = sn.feature_linear_trans_old(U, c1, use_bias=True)
        # [batch, n0, c1]

    if keep=='':
        with tf.variable_scope('linear') as sc:
            X2 = sn.flt(U, c1, is_training, bn_decay, use_bias=False)
            #X2 = sn.feature_linear_trans_old(U, c1, use_bias=False)
        # [batch, n0, c1]

    if keep=='only_lap':
        X = X1
    elif keep=='only_linear':
        X = X2
    else:
        assert(keep=='')
        X = X1 + X2

    # tf.nn.relu(X)
    return X


'''
def recurrent_operator_module(S, A, k_in, k_out):

    import surf_basic as sb

    sb.batch_diaginv()

    return
'''


'''
def fmap(DE, W1, W2, W3, W4):
    def local_linear_trans(fun, out_dim, W, use_bias=True):
        # [batch, x, in_dim]
        [batch, x, in_dim] = fun.get_shape().as_list()

        # use weights from external
        # W = weight_variable([in_dim, out_dim])

        tmp = tf.tile(tf.expand_dims(tf.expand_dims(W, 0), 0), [batch, x, 1, 1])

        # convert both to the size of [batch, x, in_dim, out_dim]
        mul = tf.multiply(tf.expand_dims(fun, -1), tmp)

        out_fun = tf.reduce_sum(mul, axis=[2])

        if use_bias:
            b = sb.bias_variable([1, 1, out_dim])
            out_fun = out_fun + b

        # [batch, x, out_dim]
        return out_fun

    # [batch, f, 3]
    [batch, f, _] = DE.get_shape().as_list()

    F0 = DE  # F0 = tf.reshape(DE, [batch, f, 3])

    F1 = local_linear_trans(F0, 1, W1, use_bias=False)

    F2 = tf.nn.relu(F1)

    F3 = local_linear_trans(F2, 10, W2, use_bias=True)

    F4 = tf.nn.relu(F3)

    F5 = local_linear_trans(F4, 10, W3, use_bias=True)

    F6 = tf.nn.relu(F5)

    F7 = local_linear_trans(F6, 1, W4, use_bias=True)

    F8 = tf.nn.relu(F7)
    # [batch, f, 1]

    return tf.reshape(F8, [batch, f])

E = per_face_edges_length(V, F)

A = tf.stack([
    angle_from_edges(E[:, :, 1], E[:, :, 2], E[:, :, 0]),
    angle_from_edges(E[:, :, 2], E[:, :, 0], E[:, :, 1]),
    angle_from_edges(E[:, :, 0], E[:, :, 1], E[:, :, 2])], axis=2)

epsilon = 1e-10

A = tf.reciprocal(tf.tan(A) + epsilon)

# now learn a 9 --> 1 mapping: fmap

# in this way, the weights are shared in all fmaps.
W1 = sb.weight_variable([3, 10])
# W1 = tf.expand_dims( tf.constant([0.0,0.0,1.0]), -1)
W2 = sb.weight_variable([10, 10])
W3 = sb.weight_variable([10, 10])
W4 = sb.weight_variable([10, 1])
'''


def assembler_20x_0(vc0, vc1, vc2):

    import surf_dg as sd

    w = sb.weight_variable([1])

    return sd.lap_assembler_v0(vc0, vc1, vc2) * tf.nn.relu(0.0001+w) + sd.mass_assembler_v0(vc0, vc1, vc2)


def assembler_20x_1(vc0, vc1, vc2):
    # input: [batch, f, v_c] * 3

    [batch, f, c] = vc0.get_shape().as_list()

    import surf_dg as sd

    def local_linear_trans(fun, out_dim, W, use_bias=True):
        # [batch, x, in_dim]
        [batch, x, in_dim] = fun.get_shape().as_list()

        # use weights from external
        # W = weight_variable([in_dim, out_dim])

        tmp = tf.tile(tf.expand_dims(tf.expand_dims(W, 0), 0), [batch, x, 1, 1])

        # convert both to the size of [batch, x, in_dim, out_dim]
        mul = tf.multiply(tf.expand_dims(fun, -1), tmp)

        out_fun = tf.reduce_sum(mul, axis=[2])

        if use_bias:
            b = sb.bias_variable([1, 1, out_dim])
            out_fun = out_fun + b

        # [batch, x, out_dim]
        return out_fun

    W1 = sb.weight_variable([3*c, 3*c])
    W2 = sb.weight_variable([3*c, 3*c])
    W3 = sb.weight_variable([3*c, 3*c])
    W4 = sb.weight_variable([3*c, 3*c])

    F0 = tf.concat([vc0, vc1, vc2], axis=2)  # F0 = tf.reshape(DE, [batch, f, 3])
    # [batch, f, v_c * 3]

    F1 = local_linear_trans(F0, 3*c, W1, use_bias=False)

    F2 = tf.nn.relu(F1)

    F3 = local_linear_trans(F2, 3*c, W2, use_bias=True)

    F4 = tf.nn.relu(F3)

    F5 = local_linear_trans(F4, 3*c, W3, use_bias=True)

    F6 = tf.nn.relu(F5)

    F7 = local_linear_trans(F6, 3*c, W4, use_bias=True)

    F8 = tf.nn.relu(F7)

    #w = sb.weight_variable([1])

    #return sd.lap_assembler(vc0, vc1, vc2) * tf.nn.relu(0.0001 + w) + sd.mass_assembler(vc0, vc1, vc2)

    # [batch, f, 3, f_c=3]
    return tf.reshape(F8, shape=[batch, f, 3, c])


def assembler_wrapper(assembler_core, mesh):

    def assembler(v0, v1, v2):
        return assembler_core(mesh, v0, v1, v2)

    return assembler


def assembler_feature_default(mesh, VC, metadata):
    # do not use VC by default
    # this one works for lap_assembler
    return mesh.V


def assembler_mesh_lap(mesh, v0, v1, v2, metadata):
    return sn.lap_assembler_v0(v0, v1, v2)


#def assembler_feature_v1(mesh, VC, metadata):
#    sn.grad_assembler()


def assembler_feature_m(mesh, VC, metadata):
    # do not use VC by default
    # this one works for lap_assembler

    import surf_pointnet

    def flt(U, out_dim, name=None):
        return sn.flt(U, out_dim,
                   metadata.is_training, metadata.bn_decay, use_bias=True, name=name)

    def nflt(U, out_dim, name=None):
        return sn.flt(U, out_dim,
                   metadata.is_training, metadata.bn_decay, use_bias=False, name=name)

    out = flt(VC, out_dim=10, name='0') + nflt(mesh.V, out_dim=10, name='1')

    out = tf.nn.relu(flt(out, out_dim=10, name='2'))

    out = tf.nn.relu(flt(out, out_dim=10, name='3'))

    out = flt(out, out_dim=3, name='4') # no relu in the last to allow negative weights.

    return out


def assembler_mesh_m(mesh, v0, v1, v2, metadata):
    [batch, f, v_c] = sb.get_tensor_shape(v0)
    # tf.reshape(out, shape=[,3,3])
    return sn.lap_assembler_v0(v0, v1, v2)


def slicer_dir_edge(mesh):
    # This one does not give global information like positions.
    import surf_op as so
    import surf_dg as sd
    V2AFC0 = None #
    # so.operator_vertex_to_face_aug(mesh.V, mesh.F)
    DEC0 = tf.concat(
        [
         sd.per_face_dir_edges(mesh.V, mesh.F),
         ], axis=-1)

    slicer0 = sd.FEMSlicer(VC=None, DEC=DEC0, V2AFC=V2AFC0)

    return slicer0


def learn_slicer_local_0(mesh):
    # This one does not give global information like positions.
    import surf_op as so
    import surf_dg as sd
    V2AFC0 = None #
    # so.operator_vertex_to_face_aug(mesh.V, mesh.F)
    DEC0 = tf.concat(
        [
         sd.per_face_dir_edges(mesh.V, mesh.F),
         tf.expand_dims(sd.per_face_edges_length(mesh.V, mesh.F), axis=-1),
         tf.expand_dims(sd.per_face_angles(mesh.V, mesh.F), axis=-1)
         ], axis=-1)

    slicer0 = sd.FEMSlicer(VC=None, DEC=DEC0, V2AFC=V2AFC0)

    return slicer0


def learn_slicer_intrinsic_1(mesh):
    import surf_op as so
    import surf_dg as sd
    epsilon = 1e-2
    DEC0 = tf.concat(
        [
         # sd.per_face_dir_edges(mesh.V, mesh.F), # this is extrinsic, should not use it.
         tf.expand_dims(sd.per_face_edges_length(mesh.V, mesh.F), axis=-1),
         tf.expand_dims(sd.per_face_angles(mesh.V, mesh.F), axis=-1),
         tf.expand_dims(tf.reciprocal(tf.tan(sd.per_face_angles(mesh.V, mesh.F))+epsilon), axis=-1)
         ], axis=-1)

    import surf_basic as sb
    [batch, f, _] = sb.get_tensor_shape(mesh.F)

    V2AFC0 = None #tf.zeros(shape=[batch, f, 3, 1]) # This cannot be None since it will be sliced and used for MLP in learn_assembler.

    slicer0 = sd.FEMSlicer(VC=None, DEC=DEC0, V2AFC=V2AFC0)

    return slicer0


def learn_slicer_intrinsic_0(mesh):
    import surf_op as so
    import surf_dg as sd
    DEC0 = tf.concat(
        [
         # sd.per_face_dir_edges(mesh.V, mesh.F), # this is extrinsic, should not use it.
         tf.expand_dims(sd.per_face_edges_length(mesh.V, mesh.F), axis=-1),
         tf.expand_dims(sd.per_face_angles(mesh.V, mesh.F), axis=-1)
         ], axis=-1)

    import surf_basic as sb
    [batch, f, _] = sb.get_tensor_shape(mesh.F)

    V2AFC0 = None #tf.zeros(shape=[batch, f, 3, 1]) # This cannot be None since it will be sliced and used for MLP in learn_assembler.

    slicer0 = sd.FEMSlicer(VC=None, DEC=DEC0, V2AFC=V2AFC0)

    return slicer0


def learn_slicer_0(mesh):
    import surf_op as so
    import surf_dg as sd
    V2AFC = so.operator_vertex_to_face_aug(mesh.V, mesh.F)
    DEC = tf.concat(
        [V2AFC,
         sd.per_face_dir_edges(mesh.V, mesh.F),
         tf.expand_dims(sd.per_face_edges_length(mesh.V, mesh.F), axis=-1),
         tf.expand_dims(sd.per_face_angles(mesh.V, mesh.F), axis=-1)
         ], axis=-1)

    slicer = sd.FEMSlicer(VC=None, DEC=DEC, V2AFC=V2AFC)

    return slicer


def learn_assembler_2(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):  # vc0, vc1, vc2
    # it will be great not to use vafc1, vafc2

    assert vafc0 is None
    assert vafc1 is None
    assert vafc2 is None

    with tf.variable_scope(nname) as sc:

        def flt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=True) # Critical to share weights for each op

        def nflt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=True)

        dim1 = 32

        import utils_fun as uf
        import utils_basic as ub

        lt00 = uf.Linear(in_dim=ub.shape(ec0[2]), out_dim=dim1)

        out = lt00.forward(ec0)

        out = nflt(ec0, out_dim=dim1, name='01') \
              + nflt(ec1, out_dim=dim1, name='02') + nflt(ec2, out_dim=dim1, name='03')

        dim2 = 32

        out = tf.nn.relu(flt(out, out_dim=dim2, name='2'))

        dim3 = 32

        out = tf.nn.relu(flt(out, out_dim=dim3, name='3'))

        out = flt(out, out_dim=out_dim, name='4') # no relu in the last to allow negative weights.

    # [batch, f, f_c=3]
    return out


def learn_assembler_1(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):  # vc0, vc1, vc2
    # it will be great not to use vafc1, vafc2

    assert vafc0 is None
    assert vafc1 is None
    assert vafc2 is None

    with tf.variable_scope(nname) as sc:

        def flt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=True) # Critical to share weights for each op

        def nflt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=True)

        dim1 = 32

        out = nflt(ec0, out_dim=dim1, name='01') \
              + nflt(ec1, out_dim=dim1, name='02') + nflt(ec2, out_dim=dim1, name='03')

        dim2 = 32

        out = tf.nn.relu(flt(out, out_dim=dim2, name='2'))

        dim3 = 32

        out = tf.nn.relu(flt(out, out_dim=dim3, name='3'))

        out = flt(out, out_dim=out_dim, name='4') # no relu in the last to allow negative weights.

    # [batch, f, f_c=3]
    return out


def learn_assembler_0(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):  # vc0, vc1, vc2
    # it will be great not to use vafc1, vafc2

    with tf.variable_scope(nname) as sc:

        def flt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=True) # Critical to share weights for each op

        def nflt(U, out_dim, name=None):
            return sn.flt(U, out_dim,
                       metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=True)

        dim1 = 32

        out = flt(vafc0, out_dim=dim1, name='00') + nflt(ec0, out_dim=dim1, name='01') \
              + nflt(ec1, out_dim=dim1, name='02') + nflt(ec2, out_dim=dim1, name='03')

        dim2 = 32

        out = tf.nn.relu(flt(out, out_dim=dim2, name='2'))

        dim3 = 32

        out = tf.nn.relu(flt(out, out_dim=dim3, name='3'))

        out = flt(out, out_dim=out_dim, name='4') # no relu in the last to allow negative weights.

    # [batch, f, f_c=3]
    return out


def grad_assembler_pos_test32_0(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):
    import surf_dg as sd
    return sd.grad_assembler_pos_test32(vafc0, vafc1, vafc2, ec0, ec1, ec2)


def grad_assembler_dir_edge_test32_0(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):
    import surf_dg as sd
    return sd.grad_assembler_dir_edge_test32(vafc0, vafc1, vafc2, ec0, ec1, ec2)


def grad_assembler_dir_edge_test48_0(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, nname):
    import surf_dg as sd
    return sd.grad_assembler_dir_edge_test48(vafc0, vafc1, vafc2, ec0, ec1, ec2)


def dirac_strong_assembler_test(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata=None, nname=None):
    assert out_dim%4 == 0

    # [batch, f, f_c=3]
    import surf_dg as sd
    D = sd.dirac_strong_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2)

    [batch, f, _] = sb.get_tensor_shape(D)
    D = tf.concat([tf.zeros(shape=[batch, f, 1]), D], axis=-1)
    # D becomes [batch, f, 4]

    return tf.concat([D for i in range(out_dim//4)], axis=-1)


def dirac_strong_assembler_test48(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata=None, nname=None):
    assert out_dim%3 == 0

    # [batch, f, f_c=3]
    import surf_dg as sd
    D = sd.dirac_strong_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2)
    return tf.concat([D for i in range(out_dim//3)], axis=-1)


def learn_assembler_wrapper(learn_assembler, out_dim, metadata, name):
    def learn_assembler_nometa(vafc0, vafc1, vafc2, ec0, ec1, ec2):
        return learn_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2, out_dim, metadata, name)
    return learn_assembler_nometa


import numpy as np
def input_fun_hks2(mesh, metadata=None):
    k = 10
    times = np.power(10, np.arange(-3, -0.5, 0.1))

    U = tf.concat([sm.generalized_heat_kernel_signature_approx(mesh, k=k, time=time, pre_cond=None) for time in times], axis=-1)

    return U



import numpy as np
def input_fun_hks(mesh, metadata=None):
    k = 5
    times = np.power(10, np.arange(-2, 0.4, 0.2))

    U = tf.concat([sm.generalized_heat_kernel_signature_approx(mesh, k=k, time=time, pre_cond=None) for time in times], axis=-1)

    return U


def input_fun_hks_load(mesh, metadata=None):

    import numpy as np
    times = np.power(10, np.arange(-2, 0.4, 0.2)) # increment is 0.2

    U = tf.concat([sm.heat_kernel_signature_from_eigen(mesh, time) for time in times], axis=-1)

    return U


