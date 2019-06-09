from surf_basic import *



def operator_vertex_to_face_aug( vertex_fun, F):

    [batch, n, v_c] = vertex_fun.get_shape().as_list()
    [_, f, dim] =  F.get_shape().as_list()
    assert(dim==3)

    face_fun = tf.stack([tf.stack([tf.gather(vertex_fun[i,:,:], F[i,:,0]),
        tf.gather(vertex_fun[i,:,:], F[i,:,1]),
        tf.gather(vertex_fun[i,:,:], F[i,:,2])], axis=1) for i in range(batch)], axis=0)

    # [batch, f, 3, v_c]

    return face_fun


def operator_vertex_to_face_aug_old( vertex_fun, F):

    [batch, n, v_c] = vertex_fun.get_shape().as_list()
    [_, f, dim] =  F.get_shape().as_list()
    assert(dim==3)

    bindex = tf.stack([i*tf.ones([f],dtype=F.dtype) for i in range(batch)])

    # assert these have the same dims
    # print(bindex.shape)
    # print(F[:,:,0].shape)

    face_node0 = tf.stack([bindex,F[:,:,0]],axis=2)
    face_node1 = tf.stack([bindex,F[:,:,1]],axis=2)
    face_node2 = tf.stack([bindex,F[:,:,2]],axis=2)

    #print('operator_vertex_to_face_aug():face_node2 ',face_node2.shape)

    face_fun0 = tf.gather_nd(vertex_fun, face_node0)
    face_fun1 = tf.gather_nd(vertex_fun, face_node1)
    face_fun2 = tf.gather_nd(vertex_fun, face_node2)
    # according to the gather_nd documentation, the last dim corresponds to v_c should have been just left there.

    # important: note cannot use function tf.gather here, since the indices for different
    # batch are different!
    # [batch, f, v_c]

    face_fun = tf.stack([face_fun0,face_fun1,face_fun2],axis=2)
    # [batch, f, 3, v_c]

    return face_fun


def operator_vertex_to_face( vertex_fun, F):
    # [batch, n, v_c]
    
    face_aug_fun = operator_vertex_to_face_aug( vertex_fun, F)
    
    [batch, f, dim, v_c] = face_aug_fun.get_shape().as_list()
    
    assert(dim==3)
    
    # [batch, n, 3*v_c]
    return tf.reshape(face_aug_fun, shape=[batch,f,dim*v_c])


# deprecated
def operator_vertex_aug_to_face_aug( aug_vertex_fun, F):
    
    # aug_vertex_fun is of dim: [batch, n, 3, v_c]
    
    # this is wrong: face_node0, face_node1, face_node2 = 
    
    [batch, f, _] = F.get_shape().as_list()

    # print('Debug')
    # tmp = tf.stack( [F[:,:,0], F[:,:,1], F[:,:,2]], axis=2)
    # print(tmp.shape) # [batch,f,3]
    
    face_fun_012 = operator_vertex_to_tuple(aug_vertex_fun, tf.stack( [F[:,:,0], F[:,:,1], F[:,:,2]], axis=2))
    face_fun_120 = operator_vertex_to_tuple(aug_vertex_fun, tf.stack( [F[:,:,1], F[:,:,2], F[:,:,0]], axis=2))
    face_fun_201 = operator_vertex_to_tuple(aug_vertex_fun, tf.stack( [F[:,:,2], F[:,:,0], F[:,:,1]], axis=2))
    # [batch, f, 3, v_c]
    
    face_tuple_fun = tf.stack([face_fun_012,face_fun_120,face_fun_201], axis=2)
    
    return face_tuple_fun
    # [batch, f, _3_, 3, v_c] # _3_ is the new dim inserted.


# deprecated    
def face_aug_aug_wise_mul(face_tuple_fun, out_dim, use_bias=True):
    
    [batch, f, dim0, dim1, v_c] = face_tuple_fun.get_shape().as_list()
    # [batch, f, 3, 3, v_c] 
    
    assert(dim0==3)
    assert(dim1==3)
    
    W = weight_variable([3,3,v_c,out_dim])
    
    tmp = tf.tile( tf.expand_dims(tf.expand_dims(W,0),0), [batch,f,1,1,1,1])
    
    # convert both to the size of [batch, f, 3, 3, v_c, xxx]
    mul = tf.multiply( tf.expand_dims(face_tuple_fun,-1), tmp)
    
    if use_bias:
        b = bias_variable( [1,1,1,1,1,out_dim] )
        mul = mul + b
    
    face_fun = tf.reduce_sum( mul, axis=[3,4])
    # [batch, f, 3, out_dim]
    
    return face_fun


def slice(A, idx, axis):
    assert (axis==2)
    shape = get_tensor_shape(A)
    if is_sparse(A):
        return tf.sparse_slice(A, start=[0,0,idx], size=[shape[0],shape[1],1])
    else:
        return A[:,:,idx]


def operator_face_aug_to_vertex_core( face_fun, J, K):
    
    # J: adjacent_faces_per_vertex
    
    # face_fun 
    # [batch, f, 3, f_c]
    [batch, f, dim, f_c] = face_fun.get_shape().as_list()
    
    assert(dim==3)
        
    # adjacent_faces_per_vertex
    # [batch, n, k]
    [_, n, k] = J.get_shape().as_list()
    [_, n2, k2] = K.get_shape().as_list()
    assert(n2==n)
    assert(k2==k)
    
    out_vertex_funs = []
    
    for i in range(k):
       
        tmp = tf.stack([i*tf.ones([n],dtype=J.dtype) for i in range(batch)])
        
        # assert these have the same dims
        # print(tmp.shape)
        # print(adjacent_faces_per_vertex[:,:,i].shape)
        
        indices_tensor = tf.stack([tmp,J[:,:,i],K[:,:,i] ],axis=2)
        
        # print(tf.gather_nd(face_fun, indices_tensor).shape) # [batch, n, f_c]
        
        out_vertex_funs.append( tf.gather_nd(face_fun, indices_tensor) )
        
        # change back the shape to size: [batch, n, 3, f_c]

    return [out_vertex_funs[i] for i in range(k)]


def operator_face_aug_to_vertex_aug( face_fun, J, K):

    list = operator_face_aug_to_vertex_core(face_fun, J, K)

    out_vertex_fun_m = tf.stack(list, axis=2)
    # [batch, n, k, f_c]

    return out_vertex_fun_m


def operator_face_aug_to_vertex(face_fun, J, K):
    list = operator_face_aug_to_vertex_core(face_fun, J, K)

    out_vertex_fun_m = tf.concat(list, axis=-1)
    # [batch, n, k*f_c]

    return out_vertex_fun_m


def operator_face_to_vertex( face_fun, J, K):
    
    # J: adjacent_faces_per_vertex
    
    [batch, f, f_c] = face_fun.get_shape().as_list()
    
    assert((f_c%3)==0)
    
    face_sep_fun = tf.reshape(face_fun, shape=[batch, f, 3, f_c//3])
    
    vertex_fun_aug = operator_face_aug_to_vertex_aug( face_sep_fun, J, K)
    # [batch, n, k, f_c//3]
    
    [_, n, k, _] = vertex_fun_aug.get_shape().as_list()
    
    vertex_fun = tf.reshape(vertex_fun_aug, shape=[batch,n,k*(f_c//3)])
    # [batch, n, k*f_c//3]
    
    return vertex_fun 


# deprecated
def vertex_aug_wise_mul(vertex_fun_m, out_v_c, use_bias=True):
    
    [batch, n, k, f_c] = vertex_fun_m.get_shape().as_list()
    # [batch, n, k, f_c]
    
    W = weight_variable2(1/k,[k,f_c,out_v_c])
    
    tmp = tf.tile( tf.expand_dims(tf.expand_dims(W,0),0), [batch,n,1,1,1])
    
    # convert both to the size of [batch, n, k, f_c, xxx]
    mul = tf.multiply( tf.expand_dims(vertex_fun_m,-1), tmp)
    
    if use_bias:
        b = bias_variable( [1,1,1,1,out_v_c] )
        mul = mul + b
    
    vertex_fun = tf.reduce_sum( mul, axis=[2,3])
    # [batch, n, out_v_c]
    
    return vertex_fun


def face_aug_wise_mul(face_fun, out_dim, use_bias=True):
    
    [batch, f, dim, f_c] = face_fun.get_shape().as_list()
    # [batch, f, 3, f_c]
    assert(dim==3)
    
    W = weight_variable2(0,[3,f_c,3,out_dim])
    
    tmp = tf.tile( tf.expand_dims(tf.expand_dims(W,0),0), [batch,f,1,1,1,1])
    
    # convert both to the size of [batch, f, 3, f_c, 3, xxx]
    mul = tf.multiply( tf.expand_dims(tf.expand_dims(face_fun,-1),-1), tmp)    
    
    out_face_fun = tf.reduce_sum( mul, axis=[2,3])
    
    if use_bias:
        b = bias_variable( [1,1,3,out_dim] )
        out_face_fun = out_face_fun + b
    
    # [batch, f, 3, out_dim]
    return out_face_fun


def feature_linear_trans(fun, out_dim, is_training, bn_decay, use_bias=True, reuse=False):

    # linear transformation + bias, batch normalization, relu.

    # fun: [batch, x, in_dim]

    import surf_pointnet as sp

    [batch, x, in_dim] = fun.get_shape().as_list()

    import tf_util
    point_feat = tf.expand_dims(fun, [2])
    # print(point_feat)
    # [B,N,1,]

    out_fun = tf_util.conv2d(point_feat, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='feature_linear_trans', bn_decay=bn_decay, reuse=reuse, use_bias=use_bias) # this used to be wrong that the bias was always added.

    out_fun = tf.squeeze(out_fun, [2])  # BxNxC

    # this bias is somehow redudant.
    #if use_bias:
    #    b = bias_variable([1, 1, out_dim])
    #    out_fun = out_fun + b

    # [batch, x, out_dim]
    return out_fun


# This is extremely memory comsuming!
def feature_linear_trans_old(fun, out_dim, use_bias=True, reuse=False):
    # [batch, x, in_dim]
    assert reuse==False # not implemented otherwise.
    return batch_linear(fun, out_dim, use_bias=use_bias)


def v2f(vertex_fun, connect):
    # [batch, n, v_c]
    # [batch, n, 3*v_c]
    return operator_vertex_to_face( vertex_fun, connect.F)


def f2v(face_fun, connect):
    # [batch, n, f_c]
    vertex_fun = operator_face_to_vertex( face_fun, connect.J, connect.K)
    # [batch, n, k*f_c//3]
    return vertex_fun


def flt(fun, out_dim, is_training, bn_decay, use_bias=True, name=None, reuse=False):
    if name is None:
        return flt_core(fun, out_dim, is_training, bn_decay, use_bias=use_bias, reuse=reuse)
    else:
        with tf.variable_scope(name) as sc:
            return flt_core(fun, out_dim, is_training, bn_decay, use_bias=use_bias, reuse=reuse)


def flt_core(fun, out_dim, is_training, bn_decay, use_bias=True, reuse=False):
    # [batch, x, in_dim]
    # [batch, x, out_dim]
    if out_dim=='same':
        [batch, x, in_dim] = get_tensor_shape(fun)
        out_dim = in_dim
    return feature_linear_trans(fun, out_dim, is_training, bn_decay, use_bias, reuse=reuse)


def channel_outer_product(A):
    [batch, x, c] = A.get_shape().as_list()
    A0 = tf.expand_dims(A,-1)
    A1 = tf.expand_dims(A,-2)
    
    return tf.multiply( A0, A1)


def channel_outer_product_compact(A):

    ACP = channel_outer_product(A)
    
    [batch, x, c0, c1] = ACP.get_shape().as_list()
    
    return tf.reshape(ACP, [batch, x, c0*c1])