from surf_basic import *
from surf_op import *

# Guide: try not to put mesh in this level of module.

def per_face_wedge(V, F):
    
    P = operator_vertex_to_face_aug( V, F)
    # P: [batch, f, 3, v_c==3]
    
    C = tf.cross( P[:,:,1,:]-P[:,:,0,:], P[:,:,2,:]-P[:,:,0,:] ) 
    
    # C: [batch, f, 3]
    return C


def per_face_normal(V, F):
    
    C = per_face_wedge(V, F)
    
    NF = tf.nn.l2_normalize(C,2,epsilon=1e-10)
    # [batch, f, 3]
    
    return NF


def _wedge_to_doublearea(wedge):
    # [batch, f]
    return tf.norm(wedge, axis=2)


def per_face_doublearea(V, F):
    
    C = per_face_wedge(V, F)
    
    DA = _wedge_to_doublearea(C)
    # [batch, f]
    
    return DA


def per_face_dir_edges(V, F):
    P = operator_vertex_to_face_aug(V, F)
    # [batch, f, 3, v_c=3]

    [batch, f, dim, v_c] = P.get_shape().as_list()

    assert (dim == 3)
    assert (v_c == 3)

    DE = tf.gather(P, [1, 2, 0], axis=2) - tf.gather(P, [2, 0, 1], axis=2)
    # this is not supported in TF yet: P[:,:,[1,2,0],:] - P[:,:,[2,0,1],:]

    # [batch, f, _3_, v_c=3]
    # _3_ is for indexing edges.
    # The order is the directed edges opposite to vertex 0, 1, 2, respectively.
    # Namely, edge 1->2, edge 2->0, edge 0->1.

    return DE


def per_face_edges_length(V, F):
    DE = per_face_dir_edges(V, F)
    # [batch, f, _3_, v_c=3]

    E = tf.norm(DE, axis=3)
    # [batch, f, _3_]
    # The order is the edges lengthes opposite to vertex 0, 1, 2, respectively.
    # Namely, edge 12, edge 20, edge 01. (or 21,02,10, since it does not matter for edge length)

    return E


def angle_from_edges(a, b, o):
    # the order of a,b does not matter.
    # o needs to be the edge opposite to the node.
    # assert a,b,o are of the same size.
    # the operation is applied element-wisely
    # the output have the same size as a,b,o.

    epsilon = 1e-10

    acos = tf.divide(tf.square(a) + tf.square(b) - tf.square(o), 2 * tf.multiply(a, b) + epsilon)

    return tf.acos(acos)


def per_face_angles(V, F):
    E = per_face_edges_length(V, F)

    A = tf.stack([
        angle_from_edges(E[:, :, 1], E[:, :, 2], E[:, :, 0]),
        angle_from_edges(E[:, :, 2], E[:, :, 0], E[:, :, 1]),
        angle_from_edges(E[:, :, 0], E[:, :, 1], E[:, :, 2])], axis=2)

    # [batch, f, 3]
    return A


# def per_vertex_gaussian_curvature(V, F):


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
            b = bias_variable([1, 1, out_dim])
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


# this is a generalization of per_face_angles
def per_face_feature(V, F):

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


    E = per_face_edges_length(V, F)

    A = tf.stack([
        angle_from_edges(E[:, :, 1], E[:, :, 2], E[:, :, 0]),
        angle_from_edges(E[:, :, 2], E[:, :, 0], E[:, :, 1]),
        angle_from_edges(E[:, :, 0], E[:, :, 1], E[:, :, 2])], axis=2)

    epsilon = 1e-10

    A = tf.reciprocal(tf.tan(A) + epsilon)

    # now learn a 9 --> 1 mapping: fmap

    # in this way, the weights are shared in all fmaps.
    W1 = weight_variable([3, 10])
    # W1 = tf.expand_dims( tf.constant([0.0,0.0,1.0]), -1)
    W2 = weight_variable([10, 10])
    W3 = weight_variable([10, 10])
    W4 = weight_variable([10, 1])

    C = tf.stack([
        fmap(tf.gather(A, [1, 2, 0], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [1, 2, 0], :]),
        fmap(tf.gather(A, [2, 0, 1], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [2, 0, 1], :]),
        fmap(tf.gather(A, [0, 1, 2], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [0, 1, 2], :])
    ], axis=2)

    # DE[:, :, 1, :], DE[:, :, 2, :], DE[:, :, 0, :]
    # DE[:, :, 2, :], DE[:, :, 0, :], DE[:, :, 1, :]
    # DE[:, :, 0, :], DE[:, :, 1, :], DE[:, :, 2, :]

    # [batch, f, 3]
    return C

'''
def per_face_feature_old(V, F):
    def angle_from_edges(a, b, o):
        # the order of a,b does not matter.
        # o needs to be the edge opposite to the node.
        # assert a,b,o are of the same size.
        # the operation is applied element-wisely
        # the output have the same size as a,b,o.

        epsilon = 1e-10

        acos = tf.divide(tf.square(a) + tf.square(b) - tf.square(o), 2 * tf.multiply(a, b) + epsilon)

        return tf.acos(acos)

    def fmap(DE):
        # [batch, f, _3_, v_c=3]

        [batch, f, _, _] = DE.get_shape().as_list()

        F0 = tf.reshape(DE, [batch, f, 9])

        F1 = feature_linear_trans(F0, 9, use_bias=True)

        F2 = tf.nn.relu(F1)

        F3 = feature_linear_trans(F2, 9, use_bias=True)

        F4 = tf.nn.relu(F3)

        F5 = feature_linear_trans(F4, 9, use_bias=True)

        F6 = tf.nn.relu(F5)

        F7 = feature_linear_trans(F6, 1, use_bias=True)

        F8 = tf.nn.relu(F7)
        # [batch, f, 1]

        return tf.reshape(F8, [batch, f])

    DE = per_face_dir_edges(V, F)
    # [batch, f, _3_, v_c=3]

    # now learn a 9 --> 1 mapping: fmap

    C = tf.stack([
        fmap(tf.gather(DE, [1, 2, 0], axis=2)),  # fmap(DE[:, :, [1, 2, 0], :]),
        fmap(tf.gather(DE, [2, 0, 1], axis=2)),  # fmap(DE[:, :, [2, 0, 1], :]),
        fmap(tf.gather(DE, [0, 1, 2], axis=2)),  # fmap(DE[:, :, [0, 1, 2], :])
    ], axis=2)

    # DE[:, :, 1, :], DE[:, :, 2, :], DE[:, :, 0, :]
    # DE[:, :, 2, :], DE[:, :, 0, :], DE[:, :, 1, :]
    # DE[:, :, 0, :], DE[:, :, 1, :], DE[:, :, 2, :]

    # [batch, f, 3]
    return C
'''


def apply_mask_perface(TF, F):
    [batch, f, _] = F.get_shape().as_list()
    mask = tf.to_float( tf.greater_equal(F[:,:,0], 0) ) # tf.ones([batch,f])
    return tf.multiply(TF, tf.expand_dims(mask, -1))


def per_face_cot(V, F):

    def cot(x):
        epsilon = 1e-10

        return tf.reciprocal(tf.tan(x) + epsilon)

    A = per_face_angles(V, F)

    C = cot(A)

    return C


def per_face_cot_thres(V, F):
    cot = per_face_cot(V, F)
    r = tf.minimum(cot, 10)
    r = tf.maximum(r, 0)
    return r


def vertex_to_face_avg(F, scalar_pervertex):
    # old interface is mesh0, mesh0.F

    [batch, n, c] = scalar_pervertex.get_shape().as_list()
    # scalar_pervertex: [batch, n, 10]

    weight_pervertex = tf.ones_like(scalar_pervertex)
    # TODO: using area based wighting rather than uniform weighting.

    scalar_pervertex_weighted = tf.multiply(scalar_pervertex, weight_pervertex)

    scalar_perface_weighted = operator_vertex_to_face(scalar_pervertex_weighted, F)
    weight_perface = operator_vertex_to_face(weight_pervertex, F)
    # scalar_perface: [batch, f, 3*10]

    [_, f, _] = scalar_perface_weighted.get_shape().as_list()

    scalar_perface_weighted = tf.reshape(scalar_perface_weighted, shape=[batch, f, 3, c])
    weight_perface = tf.reshape(weight_perface, shape=[batch, f, 3, c])

    scalar_perface_weighted = tf.reduce_sum(scalar_perface_weighted, axis=-2)
    weight_perface = tf.reduce_sum(weight_perface, axis=-2)
    # scalar_perface: [batch, n, c]

    # epsilon = 1e-10 # here no need to use it since weights are set to 1 for nonused vertices.

    return tf.divide(scalar_perface_weighted, weight_perface)


##################################################
# Requires J,K starting from here.
##################################################


def face_to_vertex_avg(DA, J, K, scalar_perface):
    # old interface is mesh0, mesh0.J, K

    #DA = per_face_doublearea(mesh0.V, mesh0.F)
    DA = tf.expand_dims(DA, axis=-1)

    scalar_perface_weight = tf.multiply(scalar_perface, DA)

    tmp = tf.squeeze( tf.stack([scalar_perface_weight, scalar_perface_weight, scalar_perface_weight], axis=2), axis=3)
    # remove singleton dimension
    #print('vertex_to_face_avg')
    #print(tmp.shape)
    seg_truth_per_tri = operator_face_to_vertex( tmp, J, K)
    weights_per_tri = operator_face_to_vertex( tf.squeeze(tf.stack([DA, DA, DA], axis=2), axis=3), J, K)

    seg_truth_per_tri = tf.reduce_sum(seg_truth_per_tri, axis=-1)
    weights_per_tri = tf.reduce_sum(weights_per_tri, axis=-1)

    epsilon = 1e-10

    return tf.divide(seg_truth_per_tri, weights_per_tri+epsilon)


def per_vertex_normal(V, F, J, K):
        
    C = per_face_wedge(V, F)
        
    C = tf.tile(tf.expand_dims( C, axis=2), [1,1,3,1])
    # [batch, f, _3_, 3] # _3_ correspondes to the new 3
        
    C = operator_face_aug_to_vertex_aug( C, J, K)
    # [batch, n, k, 3]
        
    DA = per_face_doublearea(V, F)
    
    DA = tf.expand_dims( tf.expand_dims( DA, axis=-1), axis=-1)
    
    DA = tf.tile(DA, [1,1,3,1])
    
    DA = operator_face_aug_to_vertex_aug( DA, J, K)
    # [batch, n, k, 1]
        
    epsilon = 1e-8
    
    NV = tf.reduce_sum( C, axis=2) / ( tf.reduce_sum( DA, axis=2) + epsilon)
    
    NV = tf.nn.l2_normalize(NV,2,epsilon=1e-10)

    return NV


def change_padding_value(A, padvalue, newvalue):

    #assert(padvalue==-1)
    #condition=tf.less(A, 0)

    condition = tf.equal(A, padvalue)

    R = tf.where(condition, newvalue * tf.ones_like(A), A)

    # This is wrong use of assert!: tf.assert_greater_equal(R, 0)

    return R


def succ(K):
    
    # -1 should remains -1.
    # using two conditions is really necessary.

    if False:

        # tf.batch_gather

        condition = tf.nn.relu(K) # tf.cast( tf.logical_and( tf.greater_equal( K, 0), tf.less( K+1, 3) ), dtype=tf.int32)
        K_tmp = condition + K

        condition = tf.nn.relu(K) # tf.cast( tf.logical_and( tf.greater_equal( K, 0), tf.greater_equal( K+1, 3) ), dtype=tf.int32)
        K_succ = K_tmp - 2 * condition # there must be K_tmp, cannot be K. # +1-3=-2

    else:
        # This is the old code for dense matrix.

        condition = tf.logical_and( tf.greater_equal( K, 0), tf.less( K+1, 3) )
        K_tmp = tf.where( condition, K+1, K)

        condition = tf.logical_and( tf.greater_equal( K, 0), tf.greater_equal( K+1, 3) )
        K_succ = tf.where( condition, K-2, K_tmp) # this must be K_tmp, cannot be K. # +1-3=-2
    
    return K_succ


def prev(K):
    
    # remember that the K could already be -1 for unfilled triangles. 
    
    # -1 should remains -1.
    # using two conditions is really necessary.
    
    condition = tf.logical_and( tf.greater_equal( K, 0), tf.less( K+2, 3) )
    K_tmp = tf.where( condition, K+2, K)
    
    condition = tf.logical_and( tf.greater_equal( K, 0), tf.greater_equal( K+2, 3) )
    K_prev = tf.where( condition, K-1, K_tmp) # this must be K_tmp, cannot be K. # -1+3=+2
        
    return K_prev


def mass_barycentric(V, F, J):

    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    DA = per_face_doublearea(V, F)

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]


        IM = tf.stack([bindex, J_tmp], axis=-1)
        # [batch, n, 2]

        Mi = tf.gather_nd(DA, IM)/6
        # [batch, n]

        if i==0:
            M = Mi
        else:
            M = M + Mi

    # M: [batch,n]
    return M


def mass_full(V, F, J, K, FM):

    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()

    assert (k == k2)

    DA = per_face_doublearea(V, F)
    # [batch, f]

    # not necessary according to implementation of DA.
    # C = tf.multiply(C, tf.expand_dims(FM, -1))

    return DA


'''
half_laplacian and generalized_half_laplacian now are one.
'''
def half_laplacian(V, F, J, K_succ, FM, per_face_feature_fun=per_face_cot):
    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K_succ.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    assert (k == k2)

    C = per_face_feature_fun(V, F)
    # [batch, f, 3]

    # critical: mask the unused entries.
    # C = apply_mask_perface(C, F)
    C = tf.multiply(C, tf.expand_dims(FM, -1))

    bindex = tf.stack([i * tf.ones([n], dtype=J.dtype) for i in range(batch)])

    # print(bindex.shape): [batch, n] such that bindex[i,xxx]=i


    L_list = []

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]
        K_tmp = K_succ[:, :, i]
        J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

        IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)
        # print(IM.shape): [batch, n, 3]

        Li = tf.gather_nd(C, IM)
        # print(Li.shape): [batch, n]

        L_list.append(Li)

    L = tf.stack([L_list[i] for i in range(len(L_list))], axis=-1)

    # L: [batch, n, k]
    return L

'''
def generalized_half_laplacian(V, F, J, K_succ, FM, per_face_feature_fun=per_face_feature):
    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K_succ.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    assert (k == k2)

    def cot(x):
        epsilon = 1e-10

        return tf.reciprocal(tf.tan(x) + epsilon)

    C = per_face_feature_fun(V, F)
    # [batch, f, 3]

    # critical: mask the unused entries.
    # C = apply_mask_perface(C, F)
    C = tf.multiply(C, tf.expand_dims(FM, -1))

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    # print(bindex.shape): [batch, n]


    L_list = []

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]
        K_tmp = K_succ[:, :, i]
        J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

        IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)
        # print(IM.shape): [batch, n, 3]

        Li = tf.gather_nd(C, IM)
        # print(Li.shape): [batch, n]

        L_list.append(Li)

    L = tf.stack([L_list[i] for i in range(len(L_list))], axis=-1)

    # L: [batch, n, k]
    return L
'''


# Universal FEM assembler.
# Or Generalized (learnable Dirac operator)
# Or the DEC theory.

# Mass Matrix
# U (R^1)^{fx3,n} stored as [f,3,n,c].
# for each triangle f
#   #for j in range(3), expanded for clarity.
#       i=0: U[f,i,F[f,j],0:c] = [1,0,0]
#       i=1: U[f,i,F[f,j],0:c] = [0,1,0]
#       i=2: U[f,i,F[f,j],0:c] = [0,0,1]

# Laplacian Matrix
# U (R^c)^{fx3,n} stored as [f,3,n,c=3(3d)].
# for each triangle fi
#   #for j in range(3), expanded for clarity.
#       # j: which one is the current ``major'' vertex?
#       j=0: U[fi,j,F[fi,j],0:c] = fun(1,0,0,V[F[fi,0],:], V[F[fi,1],:], V[F[fi,2],:])
#       j=1: U[fi,j,F[fi,j],0:c] = fun(0,1,0,V[F[fi,0],:], V[F[fi,1],:], V[F[fi,2],:])
#       j=2: U[fi,j,F[fi,j],0:c] = fun(0,0,1,V[F[fi,0],:], V[F[fi,1],:], V[F[fi,2],:])
# U^T*Af*U, Af is an 3f by 3f diagonal matrix of 1/3 triangle areas (repeated 3 times).
# fun is what computes the gradient. The gradients is also in the plane of triangle f.


# assembler does not have to be symmetric to its three inputs.


# FEM Matrix old
# for each triangle fi
#   vc_0 = VC[F[fi,0], :]
#   vc_1 = VC[F[fi,1], :]
#   vc_2 = VC[F[fi,2], :]
#   #for j in range(3): expanded for clarity
#       j=0: afc[fi][0] = assembler(vc_0, vc_1, vc_2) # fill in A[F[0],F[1]]
#       j=1: afc[fi][1] = assembler(vc_1, vc_2, vc_0) # fill in A[F[0],F[1]]
#       j=2: afc[fi][2] = assembler(vc_2, vc_0, vc_1) # fill in A[F[0],F[1]]
#   A[F[fi,0],F[fi,1]] = inner(afc[fi][0],afc[fi][1]) * area(fi)
#   A[F[fi,1],F[fi,0]] = same as above
#   A[F[fi,1],F[fi,2]] = inner(afc[fi][1],afc[fi][2]) * area(fi)
#   A[F[fi,2],F[fi,1]] = same as above
#   A[F[fi,2],F[fi,0]] = inner(afc[fi][2],afc[fi][0]) * area(fi)
#   A[F[fi,0],F[fi,2]] = same as above
#   A[F[fi,0],F[fi,0]] = inner(afc[fi][0],afc[fi][0]) * area(fi) * 2
#   A[F[fi,1],F[fi,1]] = inner(afc[fi][1],afc[fi][1]) * area(fi) * 2
#   A[F[fi,2],F[fi,2]] = inner(afc[fi][2],afc[fi][2]) * area(fi) * 2



def apply_fem_assembler_old(AFC, assembler):

    # AFC: [batch, f, 3, v_c]
    _, _, dim, _  = get_tensor_shape(AFC)
    assert dim==3

    AFC2 = tf.stack( [assembler(AFC[:, :, 0, :], AFC[:, :, 1, :], AFC[:, :, 2, :]),
                      assembler(AFC[:, :, 1, :], AFC[:, :, 2, :], AFC[:, :, 0, :]),
                      assembler(AFC[:, :, 2, :], AFC[:, :, 0, :], AFC[:, :, 1, :])], axis=2)

    # [batch, f, 3, f_c]
    return AFC2


def mass_assembler_old(vc0, vc1, vc2):
    # input: [batch, f, v_c] * 3

    [batch, f, v_c] = get_tensor_shape(vc0)

    # output: [batch, f, f_c]
    return tf.concat([tf.ones([batch,f,1]), tf.zeros([batch,f,2])], axis=-1)


def batch_wise_sparse_fem_old(n, F, DA, VC, assembler):

    [batch, _, _] = get_tensor_shape(F)

    ops = [[] for i in range(batch)]

    import surf_op as so
    AFC = so.operator_vertex_to_face_aug(VC, F)
    # [batch, f, 3, v_c]
    # we have vc_[{0,1,2}] = AFC[:,:,{0,1,2},:]

    # assembler: [v_c], [v_c], [v_c] -> [f_c]
    # apply three times

    AFC2 = apply_fem_assembler_old(AFC, assembler=assembler)
    # [batch, f, 3, f_c]
    # afc[fi][{0,1,2}] == AFC2[batch,fi,{0,1,2},:]

    for bi in range(batch):

        def tensor_stencil(ii,ji,s):
            def cf(A):
                return tf.reshape(A, [-1]) # return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)
            def ff(A):
                return tf.reshape(A,[-1])*s

            inner_prod = tf.reduce_sum(tf.multiply(AFC2[bi,:,ii,:],AFC2[bi,:,ji,:]), axis=-1)
            # [f]
            # print('inner_prod', inner_prod)
            # print(r[0], r[1], r[2])
            return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), tf.multiply(inner_prod, DA[bi,:]*(s/2))]

        T = [
            tensor_stencil(0, 0, 1/6), tensor_stencil(1, 1, 1/6), tensor_stencil(2, 2, 1/6),
            tensor_stencil(0, 1, 1/12), tensor_stencil(1, 2, 1/12), tensor_stencil(2, 0, 1/12),
            tensor_stencil(1, 0, 1/12), tensor_stencil(2, 1, 1/12), tensor_stencil(0, 2, 1/12)
        ]

        ops[bi] = SparseTensorWrapper(indices=tf.stack([
                            tf.concat([T[i][0] for i in range(len(T))], axis=0),
                            tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1),
                     values=tf.concat([T[i][2] for i in range(len(T))], axis=0), dense_shape=[n,n])

    return ops


# FEM Matrix
# for each triangle fi
#   vc_0 = VC[F[fi,0], :]
#   vc_1 = VC[F[fi,1], :]
#   vc_2 = VC[F[fi,2], :]
#   #for j in range(3): expanded for clarity
#       j=0: afc[fi][0] = assembler(vc_0, vc_1, vc_2) # fill in A[F[0],F[1]]: afc[fi][0] is 3xf_c
#       j=1: afc[fi][1] = assembler(vc_1, vc_2, vc_0) # fill in A[F[0],F[1]]: afc[fi][1] is 3xf_c
#       j=2: afc[fi][2] = assembler(vc_2, vc_0, vc_1) # fill in A[F[0],F[1]]: afc[fi][2] is 3xf_c
#   A[F[fi,0],F[fi,1]] = inner(afc[fi][0][0,1,2],afc[fi][1][2,0,1]) * area(fi)
#   A[F[fi,1],F[fi,0]] = same as above
#   A[F[fi,1],F[fi,2]] = inner(afc[fi][1][2,0,1],afc[fi][2][1,2,0]) * area(fi)
#   A[F[fi,2],F[fi,1]] = same as above
#   A[F[fi,2],F[fi,0]] = inner(afc[fi][2][1,2,0],afc[fi][0][0,1,2]) * area(fi)
#   A[F[fi,0],F[fi,2]] = same as above
#   A[F[fi,0],F[fi,0]] = inner(afc[fi][0][0,1,2],afc[fi][0][0,1,2]) * area(fi) * 2
#   A[F[fi,1],F[fi,1]] = inner(afc[fi][1][2,0,1],afc[fi][1][2,0,1]) * area(fi) * 2
#   A[F[fi,2],F[fi,2]] = inner(afc[fi][2][1,2,0],afc[fi][2][1,2,0]) * area(fi) * 2

#########################################################
# v0
#########################################################

def apply_fem_assembler_v0(AFC, assembler):

    # AFC: [batch, f, 3, v_c]
    _, _, dim, _  = get_tensor_shape(AFC)
    assert dim==3

    AFC2 = tf.stack( [assembler(AFC[:, :, 0, :], AFC[:, :, 1, :], AFC[:, :, 2, :]),
                      assembler(AFC[:, :, 1, :], AFC[:, :, 2, :], AFC[:, :, 0, :]),
                      assembler(AFC[:, :, 2, :], AFC[:, :, 0, :], AFC[:, :, 1, :])], axis=2) # TODO: is this a bug? Should I use 3 here?

    # [batch, f, 3, _3_, f_c] # the new _3_ is for output dim, the one added by assembler.
    return AFC2


def mass_assembler_v0(vc0, vc1, vc2):
    # The assemble always do computation in its local system, and output results in its local system.

    # The caller is responsible to do an indexing lookup before feed input to the assembler, and is
    # responsible to indexing the resutls to the right place.
    # input: [batch, f, v_c] * 3

    [batch, f, v_c] = get_tensor_shape(vc0)

    # output: [batch, f, 3, f_c] the _3_ is in the order, myself, next, nextnext=prev.
    return tf.concat([tf.ones([batch,f,1,1]), tf.zeros([batch,f,2,1])], axis=2)


def lap_assembler_v0(vc0, vc1, vc2):

    # input: [batch, f, v_c] * 3

    r = grad_core_from_pos(vc0, vc1, vc2)

    # r: [batch, f, f_c=v_c=3]

    result = tf.tile(tf.expand_dims(r, axis=2), multiples=[1,1,3,1])
    print('lap_assembler', result)
    # [batch, f, 3, f_c=3]
    return result


def lap_assembler_wrong(vc0, vc1, vc2):

    # input: [batch, f, v_c] * 3
    [batch, f, v_c] = get_tensor_shape(vc0)
    assert v_c == 3

    v20 = (vc2 - vc0)
    v10 = (vc1 - vc0)

    def dot(A,B):
        return tf.reduce_sum(tf.multiply(A,B), axis=-1)

    epsilon = 1e-18
    dot12 = dot(v20, v10)
    # note the gradient will be a piece-wise constant vector per triangle.
    r =   tf.expand_dims(tf.div(dot(v20,v20)+epsilon, dot12*dot12+epsilon), axis=-1) * v10 \
        + tf.expand_dims(tf.div(dot(v10,v10)+epsilon, dot12*dot12+epsilon), axis=-1) * v20 \
        - tf.expand_dims(tf.reciprocal(dot12+epsilon), axis=-1) * (v10 + v20)
    # r: [batch, f, f_c=v_c=3]

    result = tf.tile(tf.expand_dims(r, axis=2), multiples=[1,1,3,1])
    print('lap_assembler', result)
    # [batch, f, 3, f_c=3]
    return result


# This yields assymetric matrix #face by #vertex.
# def batch_wise_sparse_fem_rect(n, F, DA, VC, assembler):


# This yields  p.s.d. symmetric matrix.
def batch_wise_sparse_fem_v0(n, F, DA, VC, assembler):

    [batch, _, _] = get_tensor_shape(F)

    ops = [[] for i in range(batch)]

    import surf_op as so
    AFC = so.operator_vertex_to_face_aug(VC, F)
    # [batch, f, 3, v_c]
    # we have vc_[{0,1,2}] = AFC[:,:,{0,1,2},:]

    # assembler: [v_c], [v_c], [v_c] -> [3,f_c]
    # apply three times

    AFC2 = apply_fem_assembler_v0(AFC, assembler=assembler)
    # [batch, f, 3, 3, f_c]
    # afc[fi][{0,1,2}] == AFC2[batch,fi,{0,1,2},:]

    for bi in range(batch):

        def tensor_stencil(ii,ji):
            def cf(A):
                return tf.reshape(A, [-1]) # return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)
            def ff(A):
                return tf.reshape(A,[-1])

            def disind(ind):
                # this one dispatch the
                if ind==0:
                    return [0,1,2]
                elif ind==1:
                    return [2,0,1]
                else:
                    assert ind==2
                    return [1,2,0]

            # this slicing is not allow by tf:
            #inner_prod = tf.reduce_sum(tf.multiply(AFC2[bi,:,disind(ii),:],AFC2[bi,:,disind(ji),:])) / 6  + \
            #             tf.reduce_sum(tf.multiply(AFC2[bi,:,[disind(ii)[i] for i in [0,1,1,2,2,0]],:],AFC2[bi,:,[disind(ji)[i] for i in [1,0,2,1,0,2]],:])) / 12

            # this uses the quadrature rule.
            inner_prod = tf.add_n([tf.reduce_sum(tf.multiply(AFC2[bi, :, ii, disind(ii)[i], :], AFC2[bi, :, ji, disind(ji)[i], :]), axis=[-1]) for i in [0,1,2]] ) / 6 + \
                         tf.add_n([tf.reduce_sum(tf.multiply(AFC2[bi, :, ii, disind(ii)[i], :], AFC2[bi, :, ji, disind(ji)[j], :]), axis=[-1]) for i,j in zip([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2])]) / 12

            # [f]
            # print('inner_prod', inner_prod)
            # print(r[0], r[1], r[2])

            if DA is None:
                return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), inner_prod/2]  # since I use double area.
            else:
                return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), tf.multiply(inner_prod, DA[bi,:]*(1/2))] # since I use double area.

        T = [
            tensor_stencil(0, 0), tensor_stencil(1, 1), tensor_stencil(2, 2),
            tensor_stencil(0, 1), tensor_stencil(1, 2), tensor_stencil(2, 0),
            tensor_stencil(1, 0), tensor_stencil(2, 1), tensor_stencil(0, 2)
        ]

        ops[bi] = SparseTensorWrapper(indices=tf.stack([
                            tf.concat([T[i][0] for i in range(len(T))], axis=0),
                            tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1),
                     values=tf.concat([T[i][2] for i in range(len(T))], axis=0), dense_shape=[n,n])

    return ops


#########################################################
# Galerkin FEM
#########################################################


def pos_slicer(V, F):
    import surf_op as so
    V2AFC0 = so.operator_vertex_to_face_aug(V, F)
    DEC0 = V2AFC0
    slicer = FEMSlicer(VC=None, DEC=DEC0, V2AFC=V2AFC0)
    return slicer


# This yields  p.s.d. symmetric matrix.
def batch_wise_sparse_fem_galerkin(n, F, DA, slicer, assembler_galerkin):
    # see also
    # batch_wise_sparse_fem_v0
    # batch_wise_sparse_fem

    # VC: [batch, n, __]
    # DEC: [batch, f, 3, _]

    [batch, f, _] = get_tensor_shape(F)

    findex = tf.stack([i * tf.ones([n], dtype=tf.int64) for i in range(batch)])

    Hilbert = apply_fem_assembler_galerkin_from_slicer(slicer, assembler_galerkin=assembler_galerkin)
    # [batch, f, 3, 3, cdim.

    [_, _, _, _, cdim] = get_tensor_shape(Hilbert)

    # H[i=F[j,{1,2,3}],j,:] = AFC2[batch, j, {1,2,3}]

    # afc[fi][{0,1,2}] == AFC2[batch,fi,{0,1,2},:]

    # same as  batch_wise_sparse_fem until here.

    # essentially same as batch_wise_sparse_fem_v0 starting here.

    ops = [[] for i in range(batch)]

    AFC2 = Hilbert
    # [batch, f, 3, 3, f_c]
    # afc[fi][{0,1,2}] == AFC2[batch,fi,{0,1,2},:]

    for bi in range(batch):

        def tensor_stencil(ii,ji):
            def cf(A):
                return tf.reshape(A, [-1]) # return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)
            def ff(A):
                return tf.reshape(A,[-1])

            def disind(ind):
                # this one dispatch the
                if ind==0:
                    return [0,1,2]
                elif ind==1:
                    return [2,0,1]
                else:
                    assert ind==2
                    return [1,2,0]

            # this slicing is not allow by tf:
            #inner_prod = tf.reduce_sum(tf.multiply(AFC2[bi,:,disind(ii),:],AFC2[bi,:,disind(ji),:])) / 6  + \
            #             tf.reduce_sum(tf.multiply(AFC2[bi,:,[disind(ii)[i] for i in [0,1,1,2,2,0]],:],AFC2[bi,:,[disind(ji)[i] for i in [1,0,2,1,0,2]],:])) / 12

            # this uses the quadrature rule.
            # todo: check this code later.
            inner_prod = tf.add_n([tf.reduce_sum(tf.multiply(AFC2[bi, :, disind(ii)[i], ii, :], AFC2[bi, :, disind(ji)[i], ji, :]), axis=[-1]) for i in [0,1,2]] ) / 6 + \
                         tf.add_n([tf.reduce_sum(tf.multiply(AFC2[bi, :, disind(ii)[i], ii, :], AFC2[bi, :, disind(ji)[j], ji, :]), axis=[-1]) for i,j in zip([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2])]) / 12

            # [f]
            # print('inner_prod', inner_prod)
            # print(r[0], r[1], r[2])

            if DA is None:
                return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), inner_prod/2]  # since I use double area.
            else:
                return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), tf.multiply(inner_prod, DA[bi,:]*(1/2))] # since I use double area.

        T = [
            tensor_stencil(0, 0), tensor_stencil(1, 1), tensor_stencil(2, 2),
            tensor_stencil(0, 1), tensor_stencil(1, 2), tensor_stencil(2, 0),
            tensor_stencil(1, 0), tensor_stencil(2, 1), tensor_stencil(0, 2)
        ]

        indices = tf.stack([
                tf.concat([T[i][0] for i in range(len(T))], axis=0),
                tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1)
        values = tf.expand_dims(tf.concat([T[i][2] for i in range(len(T))], axis=0), axis=-1)  # create an singleton dimension for real number of 1 channel.

        if False: # the old implementation.
            ops[bi] = SparseTensorWrapper(indices=indices, values=values, dense_shape=[n,n])
        else:
            ops[bi] = FixPatternSparseTensors(indices=indices, values=values, dense_shape=[n, n])

    return ops


def mass_slicer_galerkin(V, F):
    import surf_basic as sb
    [batch, n, _] = sb.get_tensor_shape(V)

    import surf_op as so
    V2AFC0 = so.operator_vertex_to_face_aug(tf.zeros([batch, n, 1]), F)
    slicer = FEMSlicer(VC=None, DEC=None, V2AFC=V2AFC0)
    return slicer


def mass_assembler_galerkin(vafc0, vafc1, vafc2, ec0, ec1, ec2):
    # The assemble always do computation in its local system, and output results in its local system.

    # The caller is responsible to do an indexing lookup before feed input to the assembler, and is
    # responsible to indexing the resutls to the right place.
    # input: [batch, f, v_c] * 3

    [batch, f, v_c] = get_tensor_shape(vafc0)

    # output: [batch, f, 3, f_c] the _3_ is in the order, myself, next, nextnext=prev.
    return tf.concat([tf.ones([batch,f,1,1]), tf.zeros([batch,f,2,1])], axis=2)


def lap_slicer_galerkin(V, F):
    return pos_slicer(V, F)


def lap_assembler_galerkin(vafc0, vafc1, vafc2, ec0, ec1, ec2):

    # input: [batch, f, v_c] * 3

    print('lap_assembler: input', vafc0, vafc1, vafc2)

    r = grad_core_from_pos(vafc0, vafc1, vafc2)

    # r: [batch, f, f_c=v_c=3]

    result = tf.tile(tf.expand_dims(r, axis=2), multiples=[1,1,3,1])
    print('lap_assembler', result)
    # [batch, f, 3, f_c=3]
    return result


def lap_assembler_galerkin_robust(vafc0, vafc1, vafc2, ec0, ec1, ec2):

    # input: [batch, f, v_c] * 3

    print('lap_assembler: input', vafc0, vafc1, vafc2)

    r = grad_core_from_pos_robust(vafc0, vafc1, vafc2)

    # r: [batch, f, f_c=v_c=3]

    result = tf.tile(tf.expand_dims(r, axis=2), multiples=[1,1,3,1])
    print('lap_assembler', result)
    # [batch, f, 3, f_c=3]
    return result
########################################


def mass_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2):
    # The assemble always do computation in its local system, and output results in its local system.

    # The caller is responsible to do an indexing lookup before feed input to the assembler, and is
    # responsible to indexing the resutls to the right place.
    # input: [batch, f, v_c] * 3

    [batch, f, v_c] = get_tensor_shape(vafc0)

    # output: [batch, f, f_c=3] the _3_ is in the order, myself, next, nextnext=prev.
    return tf.concat([tf.ones([batch,f,1]), tf.zeros([batch,f,2])], axis=2)


def grad_slicer(V, F):
    return pos_slicer(V, F)


def grad_assembler_adjacency(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    import surf_basic as sb
    [batch, f, _] = sb.get_tensor_shape(vafc0)

    return tf.ones(shape=[batch, f, 1])


def grad_assembler_pos(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    return grad_core_from_pos(vafc0, ec1, ec2)


# slicer should provide ec as directed edges as returned by per_face_dir_edges
def dirac_strong_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    return dirac_core_from_de(ec0, ec1, ec2, strong=True)


# it is recommended to use weak, to avoid area division.
def dirac_weak_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    return dirac_core_from_de(ec0, ec1, ec2, strong=False)


def wedge_assembler(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    return tf.cross(ec1, ec2)


def dirac_core_from_de(ec0, ec1, ec2, strong=True, epsilon=1e-6):
    # see also per_face_dir_edges()
    # note per_face_dir_edges returns ec0 opposing edge to vertex 0 as v1 - v2
    # this is close-wise, minus right hand rule. So I do not need the minus sign in dirac paper.

    if strong:
        area =  0.5*tf.expand_dims(_wedge_to_doublearea(tf.cross(ec1, ec2)), axis=-1)
        # [batch, f, 1]
        return ec0 / (area + epsilon)
    else:
        return ec0


def grad_assembler_pos_test32(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    r = tf.concat([grad_assembler_pos(vafc0, vafc1, vafc2, ec0, ec1, ec2) for ii in range(11)], axis=-1)

    # [batch, f, f_c=32]

    return r[:,:,0:32]


def grad_assembler_dir_edge(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    # [batch, f, f_c=3]
    return grad_from_dir_edge(vafc0, ec1, ec2)


def grad_assembler_dir_edge_test32(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    r = tf.concat([grad_assembler_dir_edge(vafc0, vafc1, vafc2, ec0, ec1, ec2) for ii in range(11)], axis=-1)

    # [batch, f, f_c=32]

    return r[:,:,0:32]


def grad_assembler_dir_edge_test48(vafc0, vafc1, vafc2, ec0, ec1, ec2): # vc0, vc1, vc2

    r = tf.concat([grad_assembler_dir_edge(vafc0, vafc1, vafc2, ec0, ec1, ec2) for ii in range(16)], axis=-1)

    # [batch, f, f_c=48]

    return r


def grad_from_dir_edge(ec0, ec1, ec2):

    # [batch, f, ...]

    # See: per_face_dir_edges()

    # ec0: v1 - v2
    # ec1: v2 - v0
    # ec2: v0 - v1

    v10 = -ec2[:,:,0:3]
    v20 = ec1[:,:,0:3]

    return _core(v10, v20)


def grad_core_from_pos(vc0, vc1, vc2):

    # input: [batch, f, v_c] * 3
    [batch, f, v_c] = get_tensor_shape(vc0)
    assert v_c == 3

    v20 = (vc2 - vc0)
    v10 = (vc1 - vc0)

    return _core(v10, v20, epsilon=1e-9)


def grad_core_from_pos_robust(vc0, vc1, vc2):

    # input: [batch, f, v_c] * 3
    [batch, f, v_c] = get_tensor_shape(vc0)
    assert v_c == 3

    v20 = (vc2 - vc0)
    v10 = (vc1 - vc0)

    epsilon = 5e-3 * tf.minimum(
        tf.reduce_sum(tf.abs(v20), axis=-1),
        tf.reduce_sum(tf.abs(v10), axis=-1))
    epsilon = tf.maximum(epsilon, 1e-6)
    # [batch, f]

    return _core(v10, v20, epsilon=epsilon)


def _core(v10, v20, epsilon=1e-9):

    # this computes gradient of the dirac function centered at vertex 0
    # in this triangle. The gradient is extrinsic in the 3d space.

    def dot(A,B):
        return tf.reduce_sum(tf.multiply(A,B), axis=-1)

    dot12 = dot(v20, v10)
    ns12 = tf.norm(tf.cross(v20, v10), ord=2, axis=-1)
    ns12 = ns12 * ns12
    epsilon = epsilon * epsilon
    # note the gradient will be a piece-wise constant vector per triangle.
    r =   tf.expand_dims(tf.div(dot(v20,v20)+epsilon, ns12+epsilon), axis=-1) * v10 \
        + tf.expand_dims(tf.div(dot(v10,v10)+epsilon, ns12+epsilon), axis=-1) * v20 \
        - tf.expand_dims(tf.div(dot12+epsilon, ns12+epsilon), axis=-1) * (v10 + v20)
    # r: [batch, f, f_c=v_c=3]

    return r


def apply_fem_assembler(V2AFC, DEC, assembler):

    # V2AFC: [batch, f, 3, v_c]
    # DEC:   [batch, f, 3, __] directed edge features.

    _, _, dim, _  = get_tensor_shape(V2AFC)
    assert dim==3

    AFC2 = tf.stack( [assembler(V2AFC[:, :, 0, :], V2AFC[:, :, 1, :], V2AFC[:, :, 2, :], DEC[:, :, 0, :], DEC[:, :, 1, :], DEC[:, :, 2, :]),
                      assembler(V2AFC[:, :, 1, :], V2AFC[:, :, 2, :], V2AFC[:, :, 0, :], DEC[:, :, 1, :], DEC[:, :, 2, :], DEC[:, :, 0, :]),
                      assembler(V2AFC[:, :, 2, :], V2AFC[:, :, 0, :], V2AFC[:, :, 1, :], DEC[:, :, 2, :], DEC[:, :, 0, :], DEC[:, :, 1, :])], axis=2)

    # [batch, f, _3_, f_c] # the new _3_ is for output dim, the one added by assembler.
    return AFC2


def apply_fem_assembler_galerkin_from_slicer(slicer, assembler_galerkin):
    # V2AFC: [batch, f, 3, v_c]
    # DEC:   [batch, f, 3, __] directed edge features.

    AFC2 = tf.stack( [assembler_galerkin(slicer.V2AFC0, slicer.V2AFC1, slicer.V2AFC2, slicer.DEC0, slicer.DEC1, slicer.DEC2),
                      assembler_galerkin(slicer.V2AFC1, slicer.V2AFC2, slicer.V2AFC0, slicer.DEC1, slicer.DEC2, slicer.DEC0),
                      assembler_galerkin(slicer.V2AFC2, slicer.V2AFC0, slicer.V2AFC1, slicer.DEC2, slicer.DEC0, slicer.DEC1)], axis=3) # TODO: is this a bug? Should I use 3 here?

    # [batch, f, 3, _3_, f_c] # the new _3_ is for output dim, the one added by assembler.
    return AFC2


def apply_fem_assembler_from_slicer(slicer, assembler):

    # V2AFC: [batch, f, 3, v_c]
    # DEC:   [batch, f, 3, __] directed edge features.

    AFC2 = tf.stack( [assembler(slicer.V2AFC0, slicer.V2AFC1, slicer.V2AFC2, slicer.DEC0, slicer.DEC1, slicer.DEC2),
                      assembler(slicer.V2AFC1, slicer.V2AFC2, slicer.V2AFC0, slicer.DEC1, slicer.DEC2, slicer.DEC0),
                      assembler(slicer.V2AFC2, slicer.V2AFC0, slicer.V2AFC1, slicer.DEC2, slicer.DEC0, slicer.DEC1)], axis=2)

    # [batch, f, _3_, f_c] # the new _3_ is for output dim, the one added by assembler.
    return AFC2


class FEMSlicer():
    def __init__(self, VC, DEC, V2AFC=None):

        #if V2AFC is None:
        #    import surf_op as so
        #    V2AFC = so.operator_vertex_to_face_aug(VC, F)
        #else:
        #    assert VC is None

        assert VC is None

        # [batch, f, 3, v_c]
        # we have vc_[{0,1,2}] = AFC[:,:,{0,1,2},:]

        # assembler: [v_c], [v_c], [v_c] -> [3,f_c]
        # apply three times

        if V2AFC is None:
            self.V2AFC0 = None
            self.V2AFC1 = None
            self.V2AFC2 = None

        else:
            self.V2AFC0 = V2AFC[:, :, 0, :]
            self.V2AFC1 = V2AFC[:, :, 1, :]
            self.V2AFC2 = V2AFC[:, :, 2, :]

        if DEC is None:
            self.DEC0 = None
            self.DEC1 = None
            self.DEC2 = None

        else:
            self.DEC0 = DEC[:, :, 0, :]
            self.DEC1 = DEC[:, :, 1, :]
            self.DEC2 = DEC[:, :, 2, :]

        # Hilbert = apply_fem_assembler(V2AFC, DEC, assembler=assembler)
        # [batch, f, 3, f_c]

        return


#def mlp_lap(vafc0, vafc1, vafc2, ec0, ec1, ec2):

    # afc is the vertex feature after naively copied to faces.

    # vc: [batch, n, v_c] -> [batch, f, 3, v_c]
    # ec: [batch, f, 3, c2]


    # output dimension is [batch, f, c_out]


# This yields  assymmetric matrix.
# [batch][f,n][cdim] if transpose is False, i.e. input signal is n dim-
# [batch][n,f][cdim] if transpose is True, i.e. input signal is f dim-
def batch_wise_sparse_fem(n, F, slicer, assembler, transpose=False, use_old_tensor_rep = False):

    # VC: [batch, n, __]
    # DEC: [batch, f, 3, _]

    [batch, f, _] = get_tensor_shape(F)

    findex = tf.stack([i * tf.ones([n], dtype=tf.int64) for i in range(batch)])

    Hilbert = apply_fem_assembler_from_slicer(slicer, assembler=assembler)
    # [batch, f, 3, cdim

    [_, _, _, cdim] = get_tensor_shape(Hilbert)

    # H[i=F[j,{1,2,3}],j,:] = AFC2[batch, j, {1,2,3}]

    # afc[fi][{0,1,2}] == AFC2[batch,fi,{0,1,2},:]


    if True:
        assert not use_old_tensor_rep

        ops = [[[] for i in range(cdim)] for i in range(batch)]

        def cf(A):
            return tf.reshape(A, [-1])  # return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)

        def ff(A):
            return tf.reshape(A, [-1])

        for bi in range(batch):

            if bi==0:
                indices0 = tf.concat([tf.range(f) for ii in range(3)], axis=0)  # do it only once.
            indices1 = tf.concat([cf(F[bi, :, ii]) for ii in range(3)], axis=0)
            # values = [ tf.concat([Hilbert[bi, :, ii, cc] for ii in range(3)], axis=0) for cc in range(cdim)]
            # This works but comsuming too much: values = tf.stack([tf.concat([Hilbert[bi, :, ii, cc] for ii in range(3)], axis=0) for cc in range(cdim)], axis=-1)
            # This is wrong with critical bugs!!: values = tf.reshape(Hilbert[bi, :, :, :], shape=[3*f, cdim]) #
            values = tf.reshape(tf.transpose(Hilbert[bi, :, :, :], perm=[1,0,2]), shape=[3 * f, cdim])


            if not transpose:
                indices = tf.stack([indices0,indices1], axis=-1)
                ops[bi] = FixPatternSparseTensors(indices=indices, values=values, dense_shape=[f, n])
            else:
                indices = tf.stack([indices1, indices0], axis=-1)
                ops[bi] = FixPatternSparseTensors(indices=indices, values=values, dense_shape=[n, f])

    else:

        ops = [[[] for i in range(cdim)] for i in range(batch)]

        for bi in range(batch):

            for cc in range(cdim):

                def tensor_stencil(ii):
                    def cf(A):
                        return tf.reshape(A, [-1]) # return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)
                    def ff(A):
                        return tf.reshape(A,[-1])

                    return [tf.range(f), cf(F[bi, :, ii]), Hilbert[bi, :, ii, cc]] # here assume the final output dim of Hilbert is 1

                T = [
                    tensor_stencil(0), tensor_stencil(1), tensor_stencil(2),
                ]

                if transpose==False:
                    if use_old_tensor_rep:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack([
                                        tf.concat([T[i][0] for i in range(len(T))], axis=0),
                                        tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1),
                                 values=tf.concat([T[i][2] for i in range(len(T))], axis=0), dense_shape=[f,n])
                    else:
                        ops[bi][cc] = CompositeSparseTensor()

                        values = tf.concat([T[i][2] for i in range(len(T))], axis=0)

                        if cc==0:
                            indices = tf.stack([
                                tf.concat([T[i][0] for i in range(len(T))], axis=0),
                                tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1)
                            ops[bi][cc].create_new(indices=indices, values=values, dense_shape=[f, n])
                        else:
                            ops[bi][cc].create_from(cst=ops[bi][0], values=values, dense_shape=[f, n])

                else:
                    # transpose the dims [f,n] for [n,f]
                    if use_old_tensor_rep:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack([
                                tf.concat([T[i][1] for i in range(len(T))], axis=0),
                                tf.concat([T[i][0] for i in range(len(T))], axis=0)], axis=-1),
                            values=tf.concat([T[i][2] for i in range(len(T))], axis=0), dense_shape=[n,f])
                    else:
                        ops[bi][cc] = CompositeSparseTensor()

                        values = tf.concat([T[i][2] for i in range(len(T))], axis=0)
                        if cc==0:
                            indices = tf.stack([
                                tf.concat([T[i][1] for i in range(len(T))], axis=0),
                                tf.concat([T[i][0] for i in range(len(T))], axis=0)], axis=-1)
                            ops[bi][cc].create_new(indices=indices, values=values, dense_shape=[n,f])
                        else:
                            ops[bi][cc].create_from(cst=ops[bi][0], values=values, dense_shape=[n,f])

    return ops


def sparse_tensors_trans_apply(ops, U):
    # U: [batch, n, udim] or [batch, f, udim]

    # [batch, _, _] = sd.get_tensor_shape(U)

    batch = len(ops)
    return tf.stack([ops[bi].trans_apply_to(U[bi, :, :]) for bi in range(batch)], axis=0)


def sparse_tensors_apply(ops, U, use_old_tensor_rep, keep_dim=False):
    # U: [batch, n, udim] or [batch, f, udim]

    # [batch, _, _] = sd.get_tensor_shape(U)

    batch = len(ops)
    # cc = 0
    # return tf.stack([tf.sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=U[bi,:,:]) for bi in range(batch)], axis=0)

    if True:

        assert not use_old_tensor_rep

        if keep_dim:
            return tf.stack([ops[bi].apply_to2(U[bi, :, :]) for bi in range(batch)], axis=0)
        else:
            return tf.stack([ ops[bi].apply_to(U[bi, :, :]) for bi in range(batch)], axis=0)

    else:
        assert False # This is old code no longer maintained.
        '''
        # [batch, n/f, udim * cdim]
        if use_old_tensor_rep:
            return tf.stack([
                                tf.concat([
                                              tf.sparse_tensor_dense_matmul(sp_a=ops[bi][cc], b=U[bi, :, :])
                                              for cc in range(cdim)], axis=-1)
                                for bi in range(batch)], axis=0)
        else:
            return tf.stack([
                                tf.concat([
                                              ops[bi][cc].apply_to(U[bi, :, :])  # This is the only difference.
                                              for cc in range(cdim)], axis=-1)
                                for bi in range(batch)], axis=0)
        '''


def sparse_tensors_convert_to_dense(ops, use_old_tensor_rep):
    batch = len(ops)
    cdim = len(ops[0])

    if use_old_tensor_rep:
        dense = tf.stack([
                                  tf.stack([
                                               tf.sparse_tensor_to_dense(ops[bi][cc])
                                               for cc in range(cdim)], axis=-1)
                                  for bi in range(batch)], axis=0)  # , validate_indices=False
    else:
        dense = tf.stack([
                                  tf.stack([
                                               ops[bi][cc].to_dense()  # This is the only difference.
                                               for cc in range(cdim)], axis=-1)
                                  for bi in range(batch)], axis=0)

    # [batch, ...,..., cdim]
    return dense


#########################################################


def batch_wise_sparse_laplacian(n, F, FC):

    # FC: face feature [b,n,c]

    [batch, _, _] = get_tensor_shape(F)

    ops = [[] for i in range(batch)]

    for bi in range(batch):

        def tensor_stencil(ii,ji,vi,scale=None):
            def cf(A):
                return tf.reshape(A, [-1])# return tf.cast(tf.reshape(A,[-1]), dtype=tf.int64)
            def ff(A):
                if scale is None:
                    return tf.reshape(A*0.5,[-1])
                else:
                    return tf.reshape(A*(0.5*scale),[-1])

            return [cf(F[bi,:,ii]), cf(F[bi,:,ji]), ff(FC[bi,:,vi])]

        T = [
                tensor_stencil(0, 1, 2), tensor_stencil(1, 2, 0), tensor_stencil(2, 0, 1),
                tensor_stencil(1, 0, 2), tensor_stencil(2, 1, 0), tensor_stencil(0, 2, 1),
                tensor_stencil(0, 0, 2, scale=-1), tensor_stencil(1, 1, 0, scale=-1), tensor_stencil(2, 2, 1, scale=-1),
                tensor_stencil(1, 1, 2, scale=-1), tensor_stencil(2, 2, 0, scale=-1), tensor_stencil(0, 0, 1, scale=-1)
        ]

        ops[bi] = SparseTensorWrapper(indices=tf.stack([
                            tf.concat([T[i][0] for i in range(len(T))], axis=0),
                            tf.concat([T[i][1] for i in range(len(T))], axis=0)], axis=-1),
                     values=tf.concat([T[i][2] for i in range(len(T))], axis=0), dense_shape=[n,n])

        '''  This is slower:
        def tensor(ii,ji,vi,scale=None):
            def cf(A):
                return tf.cast(tf.reshape(A,[-1]))
            if scale is None:
                vv = cf(FC[bi,:,vi])
            else:
                vv = cf(FC[bi,:,vi]*scale)
            return SparseTensorWrapper(indices=tf.stack([
                                                cf(F[bi,:,ii]),
                                                cf(F[bi,:,ji]) ],axis=-1),
                                            values=vv, dense_shape=[n,n])

        ops[bi] = tf.sparse_add(
                        tf.sparse_add(tf.sparse_add(tensor(0, 1, 2), tf.sparse_add(tensor(1, 2, 0), tensor(2, 0, 1))),
                                      tf.sparse_add(tensor(1, 0, 2), tf.sparse_add(tensor(2, 1, 0), tensor(0, 2, 1)))
                                      ),
                        tf.sparse_add(tf.sparse_add(tensor(0, 0, 2, scale=-1), tf.sparse_add(tensor(1, 1, 0, scale=-1), tensor(2, 2, 1, scale=-1))),
                                      tf.sparse_add(tensor(1, 1, 2, scale=-1), tf.sparse_add(tensor(2, 2, 0, scale=-1), tensor(0, 0, 1, scale=-1)))
                                      ),
                    )
        '''
    return ops


def sparse_laplacian(n, F, J, FC):

    # FC: face feature [b,n,c]

    [batch,f,_] = get_tensor_shape(F)
    bfindex = tf.stack([i * tf.ones([f], dtype=J.dtype) for i in range(batch)])

    return tf.add_n([SparseTensorWrapper(indices=tf.stack([ tf.reshape(bfindex,[-1]),
                                               tf.reshape(F[:,:,(0+i)%3],[-1]),
                                               tf.reshape(F[:,:,(1+i)%3],[-1])],axis=-1),
                                        values=tf.reshape(FC[:,:,(2+i)%3],[-1]), dense_shape=[batch,n,n]) for i in range(3)])


'''
def half_laplacian(V, F, J, K_succ, FM):
    # this implement cot(\alpha)
    # note: this is not 0.5*cot(\alpha)!
    # the complete fem laplacian is 0.5*cot(\alpha)+0.5*cot(\beta).
    
    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K_succ.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    assert(k==k2)

    def cot(x):
        
        epsilon = 1e-10
        
        return tf.reciprocal( tf.tan(x) + epsilon )
    
    A = per_face_angles( V, F)

    C = cot( A)
    # [batch, f, 3]
    
    # critical: mask the unused entries, since cot 0 is not 0!
    # C = apply_mask_perface(C, F)
    C = tf.multiply(C, tf.expand_dims(FM, -1))
        
    bindex = tf.stack([i*tf.ones([n],dtype=tf.int32) for i in range(batch)])
        
    # print(bindex.shape): [batch, n]
    
    
    L_list = []
    
    for i in range(k):
        
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:,:,i]
        K_tmp = K_succ[:,:,i]
        J_tmp = tf.where( tf.less(J_tmp,0), (f-1)*tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where( tf.less(K_tmp,0), (3-1)*tf.ones_like(K_tmp), K_tmp)
        
        IM = tf.stack([bindex,J_tmp,K_tmp],axis=-1)
        # print(IM.shape): [batch, n, 3]
        
        Li = tf.gather_nd(C, IM)
        # print(Li.shape): [batch, n]
        
        L_list.append(Li)
    
    L = tf.stack([L_list[i] for i in range(len(L_list))], axis=-1)
    
    # L: [batch, n, k]
    return L
'''

def half_laplacian_from_edge(V, F, J, K_succ, FM, E):
    [batch, n, _] = V.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K_succ.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    assert (k == k2)

    def cot(x):
        epsilon = 1e-10

        return tf.reciprocal(tf.tan(x) + epsilon)

    # E = per_face_edges_length(V, F)

    A = tf.stack([
        angle_from_edges(E[:, :, 1], E[:, :, 2], E[:, :, 0]),
        angle_from_edges(E[:, :, 2], E[:, :, 0], E[:, :, 1]),
        angle_from_edges(E[:, :, 0], E[:, :, 1], E[:, :, 2])], axis=2)

    C = cot(A)
    # [batch, f, 3]

    # critical: mask the unused entries.
    # C = apply_mask_perface(C, F)
    C = tf.multiply(C, tf.expand_dims(FM, -1))

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    # print(bindex.shape): [batch, n]


    L_list = []

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]
        K_tmp = K_succ[:, :, i]
        J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

        IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)
        # print(IM.shape): [batch, n, 3]

        Li = tf.gather_nd(C, IM)
        # print(Li.shape): [batch, n]

        L_list.append(Li)

    L = tf.stack([L_list[i] for i in range(len(L_list))], axis=-1)

    # L: [batch, n, k]
    return L

'''
def laplacian_smoothing_pre_1D(u, F, J, K, b_prev=True):

    [batch, n] = u.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K.get_shape().as_list()
    assert(k==k2)
    
    if b_prev:
        K_prev = prev( K)
    else:
        K_prev = succ( K)
        
    J = J # TODO: somehow it does not work for J with -1 term..., fix it later.

    bindex = tf.stack([i*tf.ones([n],dtype=tf.int32) for i in range(batch)])
    bfindex = tf.stack([i*tf.ones([f],dtype=tf.int32) for i in range(batch)])
    
    FJK_list = []
    
    u_list= []
    
    for i in range(k):
        
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:,:,i]
        K_tmp = K_prev[:,:,i]
        J_tmp = tf.where( tf.less(J_tmp,0), (f-1)*tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where( tf.less(K_tmp,0), (3-1)*tf.ones_like(K_tmp), K_tmp)
        
        IM = tf.stack([bindex,J_tmp,K_tmp],axis=-1)
                
        # print(IM.shape): [batch, n, 3]
        
        FJKi = tf.gather_nd(F, IM)
        # print(FJKi.shape)# [batch, n]
        
        # print(IM)
        
        FJK_list.append(FJKi)
        
        IM2 = tf.stack([bindex,FJKi],axis=-1)
        # print(IM2.shape)
        
        ui = tf.gather_nd(u, IM2)
        
        u_list.append(ui)
        
    return tf.stack([u_list[i] for i in range(len(u_list))], axis=-1)

# old, has not been updated for a while.
def laplacian_smoothing_1D(V, F, J, K, u):
    
    # u
    # [batch, n]
    
    L0 = half_laplacian(V, F, J, K, True)
    # [batch, n, k]  
        
    v0 = laplacian_smoothing_pre_1D(u, F, J, K, True)
    # [batch, n, k]
    
    L1 = half_laplacian(V, F, J, K, False)
    # [batch, n, k]  
        
    v1 = laplacian_smoothing_pre_1D(u, F, J, K, False)
    # [batch, n, k]    
    
    _, _, k = v0.get_shape().as_list()
    
    for i in range(k):   
        dw = ( tf.multiply( L0[:,:,i], v0[:,:,i]-u) + tf.multiply( L1[:,:,i], v1[:,:,i]-u) ) / 2
        if i==0:
            w = dw
        else:
            w = w + dw
    return w

def half_laplacian_smoothing_pre_old(U, F, J, K_prev):
    [batch, n, uc] = U.get_shape().as_list()
    [_, f, _] = F.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    [_, _, k2] = K_prev.get_shape().as_list()
    assert (k == k2)

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])
    bfindex = tf.stack([i * tf.ones([f], dtype=tf.int32) for i in range(batch)])

    FJK_list = []

    su_list = []

    for j in range(uc):

        u_list = []

        for i in range(k):
            # index of the type [0, -1, -1] is not allowed.
            J_tmp = J[:, :, i]
            K_tmp = K_prev[:, :, i]
            J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
            K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

            IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)

            # print(IM.shape): [batch, n, 3]

            FJKi = tf.gather_nd(F, IM)
            # print(FJKi.shape)# [batch, n]

            # print(IM)

            FJK_list.append(FJKi)

            IM2 = tf.stack([bindex, FJKi], axis=-1)
            # print(IM2.shape)

            ui = tf.gather_nd(U[:, :, j], IM2)

            u_list.append(ui)

        su_list.append(tf.stack([u_list[i] for i in range(len(u_list))], axis=-1))

    return tf.stack([su_list[j] for j in range(len(su_list))], axis=-1)
'''

# half_laplacian_index is half_laplacian


'''
Moved to surf_model, since very inefficient
def half_laplacian_index(F, J, K_prev):
    #[batch, n, uc] = U.get_shape().as_list()
    [batch, f, _] = F.get_shape().as_list()
    [_, n, k] = J.get_shape().as_list()
    [_, _, k2] = K_prev.get_shape().as_list()
    assert (k == k2)

    bindex = tf.stack([i * tf.ones([n], dtype=J.dtype) for i in range(batch)])

    FJK_list = []

    su_list = []

    IM2_list = []

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]
        K_tmp = K_prev[:, :, i]
        J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

        IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)

        # print(IM.shape): [batch, n, 3]

        FJKi = tf.gather_nd(F, IM)
        # print(FJKi.shape)# [batch, n]

        # print(IM)

        FJK_list.append(FJKi)

        #IM2 = tf.stack([bindex, FJKi], axis=-1)
        # print(IM2.shape)

        #IM2_list.append(IM2)

    # [batch, n, k]
    return tf.stack([FJK_list[i] for i in range(k)], axis=-1)
'''


def half_laplacian_index(F, J, K_prev, bindex):
    #[batch, n, uc] = U.get_shape().as_list()
    [batch, f, _] = F.get_shape().as_list()
    [_, n, k] = J.get_shape().as_list()
    [_, _, k2] = K_prev.get_shape().as_list()
    assert (k == k2)

    # slow to compute, moved outside
    # bindex = tf.stack([i * tf.ones([n], dtype=J.dtype) for i in range(batch)])

    FJK_list = []

    su_list = []

    IM2_list = []

    for i in range(k):
        # index of the type [0, -1, -1] is not allowed.
        J_tmp = J[:, :, i]
        K_tmp = K_prev[:, :, i]
        J_tmp = tf.where(tf.less(J_tmp, 0), (f - 1) * tf.ones_like(J_tmp), J_tmp)
        K_tmp = tf.where(tf.less(K_tmp, 0), (3 - 1) * tf.ones_like(K_tmp), K_tmp)

        IM = tf.stack([bindex, J_tmp, K_tmp], axis=-1)

        # print(IM.shape): [batch, n, 3]

        FJKi = tf.gather_nd(F, IM)
        # print(FJKi.shape)# [batch, n]

        # print(IM)

        FJK_list.append(FJKi)

        #IM2 = tf.stack([bindex, FJKi], axis=-1)
        # print(IM2.shape)

        #IM2_list.append(IM2)

    # [batch, n, k]
    return tf.stack([FJK_list[i] for i in range(k)], axis=-1)


def half_laplacian_smoothing_pre(U, F, J, K_prev):

    [batch, n, uc] = U.get_shape().as_list()
    #[_, f, _] = F.get_shape().as_list()
    [_, _, k] = J.get_shape().as_list()
    #[_, _, k2] = K_prev.get_shape().as_list()
    #assert(k==k2)

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    su_list = []

    FJK_list = half_laplacian_index(F, J, K_prev)
    
    for j in range(uc):
    
        u_list= []

        for i in range(k):

            FJKi = FJK_list[:,:,i]
            IM2 = tf.stack([bindex, FJKi], axis=-1)

            ui = tf.gather_nd(U[:,:,j], IM2)

            u_list.append(ui)

        su_list.append( tf.stack([u_list[i] for i in range(len(u_list))], axis=-1) )
         
    return tf.stack([su_list[j] for j in range(len(su_list))], axis=-1)


def half_laplacian_expand_apply(U, FJK_list):
    [batch, n, uc] = U.get_shape().as_list()
    # [_, f, _] = F.get_shape().as_list()
    #[_, _, k] = J.get_shape().as_list()
    [_, _, k] = FJK_list.get_shape().as_list()
    # [_, _, k2] = K_prev.get_shape().as_list()
    # assert(k==k2)

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    su_list = []

    #FJK_list = half_laplacian_index(F, J, K_prev)

    for j in range(uc):

        u_list = []

        for i in range(k):
            FJKi = FJK_list[:, :, i]

            IM2 = tf.stack([bindex, FJKi], axis=-1)

            ui = tf.gather_nd(U[:, :, j], IM2)
            # [batch, n]

            u_list.append(ui)

        # tmp = tf.stack([u_list[i] for i in range(len(u_list))], axis=-1)
        # tmp: [batch, n, k]
        su_list.append(tf.stack([u_list[i] for i in range(len(u_list))], axis=-1))

    # [batch, n, k, uc]
    return tf.stack([su_list[j] for j in range(len(su_list))], axis=-1)


def half_laplacian_apply(U, FJK_list, W, bindex):

    [batch, n, uc] = U.get_shape().as_list()
    # [_, f, _] = F.get_shape().as_list()
    #[_, _, k] = J.get_shape().as_list()
    [_, _, k] = FJK_list.get_shape().as_list()
    # [_, _, k2] = K_prev.get_shape().as_list()
    # assert(k==k2)

    # bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    uj_list = []

    #FJK_list = half_laplacian_index(F, J, K_prev)

    for j in range(uc):

        uj = tf.zeros_like(U[:, :, j])

        for i in range(k):
            FJKi = FJK_list[:, :, i]

            IM2 = tf.stack([bindex, FJKi], axis=-1)

            ui = tf.gather_nd(U[:, :, j], IM2)
            # [batch, n]

            uj = uj + (tf.multiply( W[:, :, i], ui - U[:, :, j])) / 2

        uj_list.append(uj)

    # [batch, n, uc]
    return tf.stack([uj_list[j] for j in range(len(uj_list))], axis=-1)


def half_laplacian_apply_old(U, FJK_list, W):

    [batch, n, uc] = U.get_shape().as_list()
    # [_, f, _] = F.get_shape().as_list()
    #[_, _, k] = J.get_shape().as_list()
    [_, _, k] = FJK_list.get_shape().as_list()
    # [_, _, k2] = K_prev.get_shape().as_list()
    # assert(k==k2)

    bindex = tf.stack([i * tf.ones([n], dtype=tf.int32) for i in range(batch)])

    uj_list = []

    #FJK_list = half_laplacian_index(F, J, K_prev)

    for j in range(uc):

        uj = tf.zeros_like(U[:, :, j])

        for i in range(k):
            FJKi = FJK_list[:, :, i]

            IM2 = tf.stack([bindex, FJKi], axis=-1)

            ui = tf.gather_nd(U[:, :, j], IM2)
            # [batch, n]

            uj = uj + (tf.multiply( W[:, :, i], ui - U[:, :, j])) / 2

        uj_list.append(uj)

    # [batch, n, uc]
    return tf.stack([uj_list[j] for j in range(len(uj_list))], axis=-1)


def laplacian_matvec_core(V, F, J, U, K_succ, K_prev, FM):
    # U
    # [batch, n, uc]

    L0 = half_laplacian(V, F, J, K_succ, FM)
    # [batch, n, k]
    V0 = half_laplacian_smoothing_pre(U, F, J, K_prev)
    # [batch, n, k]
    
    L1 = half_laplacian(V, F, J, K_prev, FM)
    # [batch, n, k]
    V1 = half_laplacian_smoothing_pre(U, F, J, K_succ)
    # [batch, n, k]    
    
    _, _, k, uc = V0.get_shape().as_list()

    assert (k>0)
    for i in range(k):   
        dW = ( tf.multiply( tf.expand_dims(L0[:,:,i],-1), V0[:,:,i,:]-U) 
              + tf.multiply( tf.expand_dims(L1[:,:,i],-1), V1[:,:,i,:]-U) ) / 2
        if i==0:
            W = dW
        else:
            W = W + dW
    return W


def approximator(V, F, J, K):
    
    P = operator_vertex_to_tuple( V, F)
    # P: [batch, f, 3, v_c==3]
    
    C = tf.cross( P[:,:,1,:]-P[:,:,0,:], P[:,:,2,:]-P[:,:,0,:] ) 
    
    # C: [batch, f, 3]
    return C


# output ops: [batch][f,3,n][c2]
# scatter will seperte the dim for [c2]: i.e. ops[batch][c2] = [f,3,n]
# aug_face will merge f,3 into one [f*3,n]
# transpose will transpose it to [n,f,3] (or [n,f*3] if aug_face==True)
def batch_wise_sparse_dirac(n, F, VF, fun, c2, scatter_dims=False, transpose=False, aug_face=True):

    # fun takes in three fxc1 matrices, and outputs one fxc2 vector.

    # VF: vertex feature [b,n,c1]

    [batch, f, _] = get_tensor_shape(F)

    # F = tf.cast(F, dtype=tf.int64)

    if scatter_dims:
        ops = [[[] for j in range(c2)] for i in range(batch)]
    else:
        ops = [[] for i in range(batch)]


    #fI = tf.range(0, limit=f, dtype=tf.int64)
    #fJ = [  tf.constant(0, shape=[f], dtype=tf.int64),
    #        tf.constant(1, shape=[f], dtype=tf.int64),
    #        tf.constant(2, shape=[f], dtype=tf.int64)]
    fI = tf.range(0, limit=f, dtype=tf.int32)
    fJ = [  tf.constant(0, shape=[f], dtype=tf.int32),
            tf.constant(1, shape=[f], dtype=tf.int32),
            tf.constant(2, shape=[f], dtype=tf.int32)]
    #fF0 = F[:, 0]
    #fF1 = F[:, 1]
    #fF2 = F[:, 2]


    import surf_op as so
    if fun == 'ones':
        FAC = []
    else:
        FAC = so.operator_vertex_to_face_aug(VF, F)
        # [batch, f, 3, c1]

    for bi in range(batch):

        def tensor_stencil(j, IV, scale=None):
            #assert(F.dtype==tf.int64)
            if aug_face:
                if not transpose:
                    return [fI+int(f)*fJ[j], F[bi,:,j], IV]
                else:
                    return [F[bi,:,j], fI+int(f)*fJ[j], IV]
            else:
                if not transpose:
                    return [fI, fJ[j], F[bi,:,j], IV]
                else:
                    return [F[bi,:,j], fI, fJ[j], IV]

        '''
        print('batch_wise_sparse_dirac:', VF)
        print('batch_wise_sparse_dirac:', F[bi, :, 0])
        # print('batch_wise_sparse_dirac:', VF[bi,F[bi,:,0],0]) # tf.slice does not support non-continuous slicing
        print('batch_wise_sparse_dirac:', tf.gather(VF[bi, :, 0],indices=F[bi,:,0]) )

        # This T does not work.
        T = [
            tensor_stencil(0, fun(VF[bi,F[bi,:,0],:], VF[bi,F[bi,:,1],:], VF[bi,F[bi,:,2],:]) ),
            tensor_stencil(1, fun(VF[bi,F[bi,:,1],:], VF[bi,F[bi,:,2],:], VF[bi,F[bi,:,0],:]) ),
            tensor_stencil(2, fun(VF[bi,F[bi,:,2],:], VF[bi,F[bi,:,0],:], VF[bi,F[bi,:,1],:]) ),
        ]
        '''

        if fun == 'ones':
            assert c2 == 1
            ones = tf.ones([f, 1])
            T = [
                tensor_stencil(0, ones ),
                tensor_stencil(1, ones ),
                tensor_stencil(2, ones ),
            ]

        else:

            T = [
                tensor_stencil(0, fun(FAC[bi,:,0,:], FAC[bi,:,1,:], FAC[bi,:,2,:]) ),
                tensor_stencil(1, fun(FAC[bi,:,1,:], FAC[bi,:,2,:], FAC[bi,:,0,:]) ),
                tensor_stencil(2, fun(FAC[bi,:,2,:], FAC[bi,:,0,:], FAC[bi,:,1,:]) ),
            ]

        if aug_face:

            indices = [tf.concat([T[i][0] for i in range(len(T))], axis=0),
                       tf.concat([T[i][1] for i in range(len(T))], axis=0)]

            if scatter_dims:
                for cc in range(c2):
                    if not transpose:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack(indices, axis=-1),
                            values=tf.concat([T[i][2][:,cc] for i in range(len(T))], axis=0), dense_shape=[f*3,n])
                        # each has dim: [f*3,n]
                    else:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack(indices, axis=-1),
                            values=tf.concat([T[i][2][:,cc] for i in range(len(T))], axis=0),
                            dense_shape=[n, f*3])
            else:
                assert False # not implemented

        else:

            indices = [tf.concat([T[i][0] for i in range(len(T))], axis=0),
                       tf.concat([T[i][1] for i in range(len(T))], axis=0),
                       tf.concat([T[i][2] for i in range(len(T))], axis=0)]

            if scatter_dims:
                for cc in range(c2):
                    if not transpose:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack(indices, axis=-1),
                                 values=tf.concat([T[i][3][:,cc] for i in range(len(T))], axis=0), dense_shape=[f,3,n])
                        # each has dim: [f,3,n]
                    else:
                        ops[bi][cc] = SparseTensorWrapper(indices=tf.stack(indices, axis=-1),
                                                   values=tf.concat([T[i][3][:, cc] for i in range(len(T))], axis=0),
                                                   dense_shape=[n, f, 3])
                        # each has dim: [n,f,3]
            else:
                assert not transpose
                ops[bi] = \
                    [indices, #tf.SparseTensor(indices=tf.stack([], axis=-1),
                    [tf.concat([T[i][3][:, cc] for i in range(len(T))], axis=0) for cc in range(c2)],# values x cc
                     [f, 3, n]]

    return ops


def dirac_inner(U):

    # U is [f,3,n,c]

    tf.reduce_sum( tf.multiply(U, U) , axis=-1)


def batch_sparse_matmul_inner(A, B):

    # given two [batch, m, r] and [batch, r, n] matrices
    # compute the mat mul of the two matrices.
    assert False
    # TODO
    return


def sparse_matmul_inner(A, B):


    return


def merge_duplicated_entries2(indices, values, dense_shape):
    # https://stackoverflow.com/questions/38233821/merge-duplicate-indices-in-a-sparse-tensor

    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(values, new_index_positions, tf.shape(unique_indices)[0])

    return unique_indices, summed_values


def merge_duplicated_entries(indices, values, dense_shape):
    # https://stackoverflow.com/questions/38233821/merge-duplicate-indices-in-a-sparse-tensor

    # This is the wrong way using tf.assert: tf.assert_non_negative(indices)

    if len(dense_shape) == 2:

        assert len(dense_shape) == 2
        # Cannot handle different dim now.

        assert indices.dtype == tf.int32

        # indices = tf.convert_to_tensor(indices,dtype=tf.int32)

        dim = dense_shape[1]

        linearized = tf.squeeze( tf.matmul(indices, [[dim], [1]]) )  # linearized = #row_index * #cols + #col_index

        if True:

            # Get the unique indices, and their positions in the array
            y, idx = tf.unique(linearized)

            # print("merge_duplicated_entries: y, idx:", y, idx)

            # Use the positions of the unique values as the segment ids to
            # get the unique values

            # values_new = tf.segment_sum(values, idx) # this is wrong, because it does not take a value of num_segments like unsorted_segment_sum.
            values_new = tf.unsorted_segment_sum(values, idx, tf.shape(y)[0])

            # Go back to N-D indices
            y = tf.expand_dims(y, 1)
            indices_new = tf.cast(tf.concat([y // dim, y % dim], axis=1), dtype=tf.int64)

        else:

            indices_new, new_index_positions = tf.unique(linearized)
            values_new = tf.unsorted_segment_sum(values, new_index_positions, tf.shape(indices_new)[0])

            # return indices_new, values_new

        # print("merge_duplicated_entries", indices_new, values_new)

        assert get_tensor_shape(indices_new)[0] == get_tensor_shape(values_new)[0] # this is doing nothing, cannot know it before actual running.

    else:
        # merge the code later.

        assert len(dense_shape) == 3
        # Cannot handle different dim now.

        assert indices.dtype == tf.int32

        # indices = tf.convert_to_tensor(indices,dtype=tf.int32)

        #dim1 = dense_shape[1]

        mt = [dense_shape[1]*dense_shape[2], dense_shape[2], 1]

        linearized = tf.squeeze(tf.matmul(indices, [[mt[0]], [mt[1]], [mt[2]]]))  # linearized = #row_index * #cols + #col_index

        if True:

            # Get the unique indices, and their positions in the array
            y, idx = tf.unique(linearized)

            # print("merge_duplicated_entries: y, idx:", y, idx)

            # Use the positions of the unique values as the segment ids to
            # get the unique values

            # values_new = tf.segment_sum(values, idx) # this is wrong, because it does not take a value of num_segments like unsorted_segment_sum.
            values_new = tf.unsorted_segment_sum(values, idx, tf.shape(y)[0])

            # Go back to N-D indices
            y = tf.expand_dims(y, 1)
            indices_new = tf.cast(tf.concat([y // mt[0], (y % mt[0]) // mt[1], (y % mt[0]) % mt[1]], axis=1), dtype=tf.int64)

        else:

            indices_new, new_index_positions = tf.unique(linearized)
            values_new = tf.unsorted_segment_sum(values, new_index_positions, tf.shape(indices_new)[0])

            # return indices_new, values_new

        # print("merge_duplicated_entries", indices_new, values_new)

        assert get_tensor_shape(indices_new)[0] == get_tensor_shape(values_new)[
            0]  # this is doing nothing, cannot know it before actual running.

    return indices_new, values_new


# This is a wrapper of tensor sparse tensor
class SparseTensor():

    def __init__(self, indices, values, dense_shape):

        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape

        #SparseTensor(indices=tf.stack(indices, axis=-1),
        #             values=tf.concat([T[i][3][:, cc] for i in range(len(T))], axis=0), dense_shape=[f, 3, n])

        self.tf = None

    def get_tf(self):

        if self.tf is None:

            # Critical: the tf.SparseTensor disallow duplcated indices.
            self.indices_new, self.values_new = merge_duplicated_entries(self.indices, self.values, self.dense_shape)

            self.tf = tf.sparse_reorder(tf.SparseTensor(indices=self.indices_new, values=self.values_new, dense_shape=self.dense_shape))

        return self.tf


#def SparseTensorWrapper(indices, values, dense_shape):
#    st = SparseTensor(indices, values, dense_shape)
#
#    return st.get_tf()


def SparseTensorWrapper(indices, values, dense_shape):

    st = SparseTensor(indices, values, dense_shape)

    return st.get_tf()


class CompositeSparseTensor():

    def __init__(self):
        return

    def create_new(self, indices, values, dense_shape):

        assert len(dense_shape)==2 # Only support sparse matrix.
        import surf_basic as sb
        self.m = dense_shape[0]
        self.n = dense_shape[1]
        self.r = sb.get_tensor_shape(values)[0]

        # I: [m,r]
        self.I = SparseTensor(indices=tf.stack([indices[:,0],tf.range(self.r)],axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.m,self.r]).get_tf()
        # V: [r,r] but saved as an [r] tensor
        self.V = values
        # J: [r,n]
        self.J = SparseTensor(indices=tf.stack([tf.range(self.r),indices[:,1]],axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.r,self.n]).get_tf()
        return

    def create_from(self, cst, values, dense_shape):
        import surf_basic as sb
        self.m = dense_shape[0]
        self.n = dense_shape[1]
        self.r = sb.get_tensor_shape(values)[0]

        assert self.m==cst.m
        assert self.n==cst.n
        assert self.r==cst.r

        self.V = values

        self.I = cst.I
        self.J = cst.J
        return

    def apply_to(self, u):
        r0 = tf.sparse_tensor_dense_matmul(self.J, u)
        r1 = tf.multiply(tf.expand_dims(self.V,axis=-1), r0)
        return tf.sparse_tensor_dense_matmul(self.I, r1)

    def to_dense(self):
        # assert False  # Not implemented.

        return self.apply_to(tf.eye(self.n))


def op_apply_to(U, I, J, V, dense_shape, old_implementation=False):
    _, udim = get_tensor_shape(U)
    # U: [n,udim]

    [m,n] = dense_shape

    r, cdim = get_tensor_shape(V)

    if old_implementation:
        r0 = tf.sparse_tensor_dense_matmul(J, U)
    else:
        ## r0 = U[self.J,:] # this does not work, slice does not support non-continuous indices, use gather instead.
        r0 = tf.gather(params=U, indices=J, axis=0)
        # print('r0',r0)
    # r0: [f,udim]

    # return tf.concat([tf.sparse_tensor_dense_matmul(self.I,
    #                                      tf.multiply(tf.expand_dims(self.Vs[:,cc], axis=-1), r0))
    #        for cc in range(self.cdim)], axis=-1)

    r1 = tf.concat([tf.multiply(tf.expand_dims(V[:, cc], axis=-1), r0) for cc in range(cdim)], axis=-1)
    # [r,cdim*udim]
    if old_implementation:
        return tf.sparse_tensor_dense_matmul(I, r1)
    else:
        return tf.scatter_nd(indices=tf.expand_dims(I,axis=-1),
                              updates=r1,
                              shape=[m,cdim*udim])
        assert False
        # TODO: this does not work, since Variable has to be initialized first.
        result = tf.Variable(tf.zeros(shape=[self.m, cdim * udim]))
        tf.scatter_add(ref=result, indices=self.I, updates=r1)
        return result
        # This amounts to a tensor of [m,cdim,udim] but reshaped to [m,cdim*udim]


def create_transposed_operator(op):

    # TODO: this stacking can be improved.
    if not hasattr(op, 'transpose'):
        op.transpose = FixPatternSparseTensors(indices=tf.stack([op.indices[:, 1], op.indices[:, 0]], axis=-1), values=op.values, dense_shape=[op.dense_shape[1], op.dense_shape[0]])


class FixPatternSparseTensors():
    # Assert cdim can be accessed externally.

    def __init__(self, indices, values, dense_shape):

        self.old_implementation = False

        assert len(dense_shape) == 2  # Only support sparse matrix.
        import surf_basic as sb
        self.m = dense_shape[0]
        self.n = dense_shape[1]
        self.r, self.cdim = get_tensor_shape(values)

        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape

        if self.old_implementation:
            # I: [m,r]
            self.I = SparseTensor(indices=tf.stack([indices[:, 0], tf.range(self.r)], axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.m, self.r]).get_tf()
        else:
            # assert False
            # TODO: See later in apply, this does not work yet due to constratins on scatter_add.
            self.I = indices[:, 0]
        # Vs: [r, cdim]  old: virtual [cdim][r,r] but saved as cdim [r] tensors.
        self.Vs = values
        if self.old_implementation:
            # J: [r,n]
            self.J = SparseTensor(indices=tf.stack([tf.range(self.r), ], axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.r, self.n]).get_tf()
        else:
            self.J = indices[:, 1]
        return

    # Note the output of apply_to and trans_apply_to
    # are flattened to [m,n*cdim] tensors and
    # [n,m*cdim] tensors, instead of [m,n,cdim] and [n,m,cdim] tensors.
    def apply_to(self, U):
        return op_apply_to(U, self.I, self.J, self.Vs, [self.m, self.n], old_implementation=False)

    def apply_to2(self, U):
        # This one outputs [m,udim,cdim].

        _, udim = get_tensor_shape(U)
        # U: [n,udim]

        if self.old_implementation:
            r0 = tf.sparse_tensor_dense_matmul(self.J, U)
        else:
            ## r0 = U[self.J,:] # this does not work, slice does not support non-continuous indices, use gather instead.
            r0 = tf.gather(params=U, indices=self.J, axis=0)
            # print('r0',r0)
        # r0: [f,udim]

        if self.old_implementation:
            return tf.stack([
                                tf.sparse_tensor_dense_matmul(self.I, tf.multiply(tf.expand_dims(self.Vs[:,cc], axis=-1), r0))
                        for cc in range(self.cdim)], axis=-1)
        else:

            return tf.stack([
                                tf.scatter_nd(indices=tf.expand_dims(self.I, axis=-1), # use a [m,1] instead of a [m,udim,1] tensor.
                                              updates=tf.multiply(tf.expand_dims(self.Vs[:, cc], axis=-1), r0), # [f,udim]
                                              shape=[self.m,udim])
                                for cc in range(self.cdim)], axis=-1)

            assert False
            # TODO: this does not work, since Variable has to be initialized first.
            results = [[] for cc in range(self.cdim)]
            for cc in range(self.cdim):
                results[cc] = tf.Variable(tf.zeros(shape=[self.m, udim]))
                tf.scatter_add(ref=results[cc], indices=self.I,
                           updates=tf.multiply(tf.expand_dims(self.Vs[:, cc], axis=-1), r0))
            return tf.stack([
                                results[cc]
                        for cc in range(self.cdim)], axis=-1)

    def trans_apply_to(self, U):

        assert False # Try to use the transposed op instead.
        # TODO: debug and test this function.

        if not hasattr(self, 'tI'):
            # tJ: [n,r]
            self.tJ = SparseTensor(indices=tf.stack([self.indices[:, 1], tf.range(self.r)], axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.n, self.r]).get_tf()
            # tI: [r,m]
            self.tI = SparseTensor(indices=tf.stack([tf.range(self.r), self.indices[:, 0]], axis=-1), values=tf.ones(shape=self.r), dense_shape=[self.r, self.m]).get_tf()

        # U: [m,udim]
        r0 = tf.sparse_tensor_dense_matmul(self.tI, U)
        # [f,udim]

        return tf.sparse_tensor_dense_matmul(self.tJ, tf.concat([tf.multiply(tf.expand_dims(self.Vs[:, cc], axis=-1), r0) for cc in range(self.cdim)], axis=-1))

    def to_dense_flatten(self):
        # [m, n*cdim]

        return self.apply_to(tf.eye(self.n))

    def to_dense(self):
        return self.apply_to2(tf.eye(self.n))
        # return tf.reshape(self.to_dense_flatten(), shape=[self.m, self.n, self.cdim])


# naive implementation:
class FixPatternSparseTensors_naive():
    def __int__(self, indices, values, dense_shape):

        self.cdim = len(values)

        self.ops = [CompositeSparseTensor() for cc in range(self.cdim)]

        for cc in range(self.cdim):

            #  self.ops[cc] = CompositeSparseTensor()

            if cc == 0:
                self.ops[cc].create_new(indices=indices, values=values, dense_shape=dense_shape)
            else:
                self.ops[cc].create_from(cst=self.ops[0], values=values, dense_shape=dense_shape)
        return

    def apply_to(self, U):
        return tf.concat([self.ops[cc].apply_to(U) for cc in range(self.cdim)], axis=-1)


def sparse_tensor_dense_matmul(sp_a, b):
    #return tf.sparse_tensor_dense_matmul(sp_a=sp_a.get_tf(), b=b)
    #print('sparse_tensor_dense_matmul():', sp_a, b)

    return tf.sparse_tensor_dense_matmul(sp_a=sp_a, b=b)


def sparse_tensor_to_dense(T):
    # return tf.sparse_tensor_to_dense(sp_input=T.get_tf()) # , validate_indices=False
    return tf.sparse_tensor_to_dense(sp_input=T) # , validate_indices=False


#class BatchSparseTensor():
    # [batch, f,3,n, c]


class LinearOperator():
    def __init__(self, matvec):
        self._matvec = matvec
        return

    def matvec(self, u):
        return self._matvec(u)


# an operator of size [batch] [f, 3, n] [c]
class DiracOperator():

    def __init__(self, ops, scatter_dims=True, tranpose=False, aug_face=True):

        self.ops = ops
        self.batch = len(ops)
        self.c = len(ops[0])
        self.tranpose = tranpose
        self.aug_face = aug_face

    def matvec(self, u):

        def is_tf_tensor(x):
            return tf.contrib.framework.is_tensor(x)

        if is_tf_tensor(u):
            ''' '''
            if self.tranpose:
                print(self.ops)
                r = [[[] for cc in range(len(self.ops[bi]))] for bi in range(len(self.ops))]
                #if self.aug_face:
                #    [n, _] = self.ops[0][0].get_shape() # [n, f*3]
                #else:
                #    [n, _, _] = self.ops[0][0].get_shape() #tf.shape(self.ops[0][0]) # this does not work somehow # [n, f, 3]
            else:
                r = [[[] for cc in range(len(self.ops[bi]))] for bi in range(len(self.ops))]
                #[_, n, _] = get_tensor_shape(u)
            #r = tf.zeros([len(self.ops), n, len(self.ops[0])])
        else:
            r = [[[] for cc in range(len(self.ops[bi]))] for bi in range(len(self.ops))]

        if self.aug_face:
            if self.tranpose:
                # u is [batch] [f, 3], [c]
                # r is [batch] [n] [c]
                # actually same
                for bi in range(len(self.ops)):
                    for cc in range(len(self.ops[bi])):
                        # ops[bi][cc]  each has dim: [n,f,3]
                        if is_tf_tensor(u):
                            b = tf.expand_dims(tf.reshape(u[bi,:,:,cc], shape=[-1]), axis=-1)
                            # print('DiracOperator:',b.shape)
                            r[bi][cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=b)
                            # print('DiracOperator:', r[bi][cc])
                        else:
                            r[bi][cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=u[bi][cc])
                if is_tf_tensor(u):
                    # each r[bi][cc] is (1024, 1)
                    r = tf.stack([ tf.concat([r[bi][cc] for cc in range(len(self.ops[bi]))], axis=-1)
                                           for bi in range(len(self.ops))], axis=0)
            else:
                # under dev
                assert False
        else:

            if self.tranpose:
                # u is [batch] [f, 3], [c]
                # r is [batch] [n] [c]
                # actually same
                for bi in range(len(self.ops)):
                    for cc in range(len(self.ops[bi])):
                        # ops[bi][cc]  each has dim: [n,f,3]
                        if is_tf_tensor(u):
                            r[bi,:,cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=u[bi,:,:,cc])
                        else:
                            r[bi][cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=u[bi][cc])
            else:
                # u is [batch] n, [c]
                # r is [batch] [f, 3] [c]
                for bi in range(len(self.ops)):
                    for cc in range(len(self.ops[bi])):
                        # ops[bi][cc]  each has dim: [f,3,n]
                        if is_tf_tensor(u):
                            r[bi][cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=u[bi,:,:,cc])
                        else:
                            r[bi][cc] = sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=u[bi][cc])

        return r

    def to_dense(self):

        return [tf.add_n([
                      sparse_tensor_to_dense(self.ops[bi][cj])
                      for cj in range(len(self.ops[bi]))])
         for bi in range(len(self.ops))]  # each one is a [f,3,n] tensor.

    def to_squared_dense(self, weights=None):

        # weights: [batch f,3]

        def square_tensor(A, w):
            # A: [f, 3, n] tensor.
            # w: [f, 3] tensor

            import surf_basic as sb

            fa, da, n = sb.get_tensor_shape(A)
            fw, dw = sb.get_tensor_shape(w)

            assert( fa==fw)
            assert( da==3)
            assert( dw==3)

            T = tf.reshape(A, shape=[-1, n]) # T is [fx3, n]
            if w is None:
                return tf.matmul(T, T, transpose_a=True)
            else:
                w3 = tf.expand_dims(tf.reshape(w, shape=[-1]), axis=-1) # w3 is [fx3,1]
                # no need to construct w3 as a diagonal due to propogation of dims.
                r = tf.multiply(w3, T)
                print('to_squared_dense:', T, r)
                return tf.matmul(T, r, transpose_a=True)

        return [tf.add_n([
                      square_tensor(sparse_tensor_to_dense(self.ops[bi][cj]), weights[bi])
                      for cj in range(len(self.ops[bi]))])
         for bi in range(len(self.ops))]


def batch_dirac_matmul(A, B, Imatvec=None):
    # given tensors A,B [batch][f,3,n][c] , multiple them batch-wisely, yielding [batch,m,n].

    assert(len(A)==len(B))
    assert(len(A[0])==len(B[0]))
    for bi in range(A):
        assert (len(A[0]) == len(A[bi]))
        assert (len(B[0]) == len(B[bi]))

    return [ tf.add_n([ batch_sparse_matmul_inner(A[bi][cc], B[bi][cc]) for cc in range(A[bi])]) for bi in range(A)]

    if Imatvec == None:
        return tf.matmul(A, B)
    else:
        # Imatvec should be [batch,r,r]
        return tf.matmul(A, Imatvec(B))


def batch_dirac_diagmul(a, B):
    # batch_matmul( batch_diag(a), B)
    # https://stackoverflow.com/questions/37904504/is-there-an-equivalent-bsxfun-in-tensorflow-as-there-in-matlab
    # Since Tensorflow does broadcasting.
    # return tf.multiply( a, B) #somehow this is still asking for:  but are 2 and 39 for 'Mul_58' (op: 'Mul') with input shapes: [2,39], [2,39,1].
    return tf.multiply(tf.expand_dims(a, axis=-1), B)


# here fun is symmetric to 0,1,2 th inputs, but in general it does not has to be the case. (?)
# how we apply fun is subject to the permutation invariance.



# U R^{fx3,n} stored as [f,3,n,1].
# Mass Matrix
# for each vertex i
#   for each adjacent triangle f=J[i,j], j=1:k
#       put myself value at idx=F[J[i,j],K[i,j]], U[f,K[i,j],i] = 1
#       put next vertex  at idn=F[J[i,j],K_succ[i,j]], U[f,K_next[i,j],i] = 0
#       put prev vertex  at idp=F[J[i,j],K_prev[i,j]], U[f,K_prev[i,j],i] = 0
#   sum the results to get u_i
# Assume the matrix at M_ij = u_i^T*u_j = U^T*M*U


# for each vertex i
#   for each adjacent triangle f=J[i,j], j=1:k
#
#       put myself value U[f,K[i,j],i]
#       put next vertex  U[f,K_next[i,j],i]
#       put prev vertex  U[f,K_prev[i,j],i]
#   sum the results to get u_i
# Assume the matrix at M_ij = u_i^T*u_j = U^T*M*U


# for each vertex i
#   for each adjacent triangle f=J[i,j], j=1:k
#       put myself value at F[J[i,j],K[i,j]]
#       put next vertex  at F[J[i,j],K_succ[i,j]]
#       put prev vertex  at F[J[i,j],K_prev[i,j]]
#                       all to be []
#   sum the results to get u_i
# Assume the matrix at M_ij = u_i^T*u_j = U^T*M*U