import surf_net as sn
import tensorflow as tf
import numpy as np
import surf_dg as sd



def wrapper_VF(fun, mesh, name):
    if not hasattr(mesh, name):
        setattr(mesh, name, fun(mesh.V, mesh.F))


# wrap_to = lambda fun: (lambda mesh, name: wrapper_VF(fun, mesh, name))
#wrap_to = lambda fun, name: (lambda mesh: wrapper_VF(fun, mesh, name))
wrap_to = lambda name: (lambda mesh: wrapper_VF(getattr(sd, name), mesh, name))


funs = ['per_face_wedge', 'per_face_normal', 'per_face_doublearea', 'per_face_dir_edges',
            'per_face_edges_length', 'per_face_angles', 'per_face_cot']


class Set():
    def __init__(self):
        for fun in funs:
            setattr(self, fun, wrap_to(fun))

set = Set()

'''
set_per_face_wedge = wrap_to('face_wedge')
set_per_face_normal = wrap_to('face_normal')
set_per_face_doublearea = wrap_to('face_doublearea')
set_per_face_dir_edges = wrap_to('face_dir_edges')
set_per_face_edges_length = wrap_to('face_edges_length')
set_per_face_angles = wrap_to('face_angles')
set_per_face_cot = wrap_to('face_cot')
'''

import surf_basic as sb

inner = sb.batch_inner_keep
batch_matmul = sb.batch_matmul
batch_trans = sb.batch_trans
batch_diagmul = sb.batch_diagmul


def init_guess(batch, n, c):
    X0 = tf.random_uniform([batch, n, c], dtype=tf.float32, seed=1234)
    # For conjugate gradient to work, we need to manually remove the null space bases from b.
    # X0: # [batch,n,1], cb: # [batch,1,1]
    C = tf.ones_like(X0)
    # X0 = X0 - tf.multiply( X0, inner(X0, C))
    return X0


def init_guess_like(B):
    [batch, n, c] = B.get_shape().as_list()
    # X0 = tf.ones( [batch,n,1], tf.float32) # cannot use constant vector, it will fail for Laplacian, since in its kernel.
    return init_guess(batch, n, c)


def cg_iteration(x, r, p, matvec, Imatvec, epsilon):
    alpha = tf.divide( inner(r,r,Imatvec=Imatvec), inner(p,matvec(p),Imatvec=Imatvec)+epsilon )
    r_old = r
    x = x + tf.multiply( alpha, p)
    r = r - tf.multiply( alpha, matvec(p))
    # if r is sufficiently small, you may exit the loop.
    # we do not do it since there are multiple channels.
    beta = tf.divide( inner(r,r,Imatvec=Imatvec), inner(r_old,r_old,Imatvec=Imatvec)+epsilon )
    p = r + tf.multiply( beta, p)

    return x, r, p


def pcg_iteration(x, r, p, z, Amatvec, Imatvec, Pmatvec, epsilon):
    alpha = tf.divide(inner(r, z, Imatvec=Imatvec), inner(p, p, Imatvec=Amatvec) + epsilon)
    r_old = r
    z_old = z
    x = x + tf.multiply(alpha, p)
    r = r - tf.multiply(alpha, Amatvec(p))
    # if r is sufficiently small, you may exit the loop.
    # we do not do it since there are multiple channels.
    z = Pmatvec(r)
    beta = tf.divide(inner(z, r, Imatvec=Imatvec), inner(z_old, r_old, Imatvec=Imatvec) + epsilon)
    p = z + tf.multiply(beta, p)

    return x, r, p, z


def conjugate_gradient( matvec, b, x0, k=10, Imatvec=None):
    # follow the implementation here:
    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    # this potential could be implemented by a RNN unit.
    
    [batch, n, c] = x0.get_shape().as_list()
    [batch2, n2, c2] = b.get_shape().as_list()
    
    assert(batch==batch2)
    assert(n==n2)
    assert(c==c2)

    r = b - matvec(x0)
    p = r
    x = x0
    
    epsilon = 0#1e-6

    i = tf.constant(0)
    cond = lambda i, x, r, p: tf.less(i, k)
    body = lambda i, x, r, p: (tf.add(i, 1),) + cg_iteration(x, r, p, matvec=matvec, Imatvec=Imatvec, epsilon=epsilon) # tuple add.

    # important to get back these values from while_loop!
    [i, x, r, p] = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[i, x, r, p]
    )

    #for i in range(k):
    #    _, x, r, p = body(i, x, r, p)

    return x


def preconditioned_conjugate_gradient(Amatvec, Pmatvec, b, x0, Imatvec=None, k=10):
    # follow the implementation here:
    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    # this potential could be implemented by a RNN unit.
    # P = M^{-1} using their notation.

    [batch, n, c] = x0.get_shape().as_list()
    [batch2, n2, c2] = b.get_shape().as_list()

    assert (batch == batch2)
    assert (n == n2)
    assert (c == c2)

    r = b - Amatvec(x0)
    z = Pmatvec(r)
    p = z
    x = x0

    '''
    print('preconditioned_conjugate_gradient():')
    print('r shape')
    print(r.get_shape())
    print('z shape')
    print(z.get_shape())
    '''

    epsilon = 0  # 1e-6

    i = tf.constant(0)
    cond = lambda i, x, r, p, z: tf.less(i, k)
    body = lambda i, x, r, p, z: (tf.add(i, 1),) + pcg_iteration(x, r, p, z, Amatvec=Amatvec, Imatvec=Imatvec, Pmatvec=Pmatvec, epsilon=epsilon) # tuple add.

    # important to get back these values from while_loop!
    [i, x, r, p, z] = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[i, x, r, p, z]
    )

    for i in range(k):
        _, x, r, p, z = body(i, x, r, p, z)

    return x


def expm_matvec( matvec, X):
    # this is the Pade approximation of matrix exponential.
    # https://en.wikipedia.org/wiki/Pad%C3%A9_approximant#Examples
    #[batch, n, k] = A.get_shape().as_list()
    RX = X + 0.5*matvec( X + 0.25*matvec(X + 1/6*matvec(X + 1/8*matvec(X + 1/10*matvec(X) ) ) ) )
    #    X - 0.5*matvec( X - 0.25*matvec(X - 1/6*matvec(X - 1/8*matvec(X - 1/10*matvec(X) ) ) ) )
    def left_matvec(X):
        LX = X - 0.5*matvec( X - 0.25*matvec(X - 1/6*matvec(X - 1/8*matvec(X - 1/10*matvec(X) ) ) ) )
        return LX

    X0 = init_guess_like(RX)

    return conjugate_gradient( left_matvec, RX, X0, k=10)


def per_vertex_normal(mesh):

    return sd.per_vertex_normal(mesh.V, mesh.F, mesh.J, mesh.K)


def laplacian_matvec(U, mesh):
    check_laplacian_dependencies(mesh)
    return sd.laplacian_matvec_core(mesh.V, mesh.F, mesh.J, U, mesh.K_succ, mesh.K_prev, mesh.FM)


def laplacian_mean_curvature(mesh):
    # return tf.stack( [laplacian_smoothing_1D(V, F, J, K, V[:,:,i]) for i in range(3)], axis=-1)
    # Thd old one: return laplacian_matvec(mesh.V, mesh)

    return generalized_laplacian_filter(mesh.V, mesh, per_face_feature_fun=sn.per_face_cot)


def gaussian_curvature(mesh):

    # [batch, n, 1]

    if not hasattr(mesh, 'diracg'):
        mesh.diracg = SparseGraphDiracOperator(mesh)

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    set_fun = getattr(set, 'per_' + 'face_angles')
    set_fun(mesh)

    pi = 3.1415926535

    u = tf.expand_dims(mesh.per_face_angles, axis=-1)
    print('gaussian_curvature:', u.shape)

    return (pi * 2 - mesh.diracg.dirac.matvec(u)) * tf.expand_dims(mesh.VM, axis=-1)


def squared_dirac_filter(mesh):
    return


def dirac_mass(mesh):
    # VF = tf.stack([mesh.V,], axis=)

    set_fun = getattr(set, 'per_' + 'face_doublearea')
    set_fun(mesh)

    import surf_basic as sb
    [batch, n, _] = sb.get_tensor_shape(mesh.V)
    VF_mass = tf.concat( [tf.ones(shape=[batch,n,1]), tf.zeros(shape=[batch,n,2])], axis=-1)
    # VF_mass[bi,vj,:] = [1,0,0] for each vertex
    print('dirac_mass:', VF_mass)

    def fun_mass(VF0, VF1, VF2):
        # VF0 [f,1]

        # [f,c=3]
        return tf.concat([VF0,VF1,VF2], axis=-1) # reordering according to index.

    c2_mass = 3

    dirac = SparseDiracOperator(mesh, VF_mass, fun_mass, c2_mass)


    return dirac


def dirac_adj(mesh):

    dirac = SparseGraphDiracOperator(mesh)

    return dirac


def dirac_lap(mesh):
    # VF = tf.stack([mesh.V,], axis=)

    set_fun = getattr(set, 'per_' + 'face_doublearea')
    set_fun(mesh)

    VF_lap = mesh.V

    epsilon = 1e-8
    eye = tf.expand_dims(tf.eye(2)*epsilon, axis=0)

    # BxNxC
    def fun_lap(VF0, VF1, VF2):
        # compute gradients in the extrinsic coordinates.

        # VF0: [f,3] each row is the position of the first vertex in the triangle.

        import surf_basic as sb
        [f, _] = sb.get_tensor_shape(VF0)

        b = - tf.ones([f,2])


        A = tf.matmul(tf.stack([VF1 - VF0, VF2 - VF0], axis=1),
                  tf.stack([VF1 - VF0, VF2 - VF0], axis=1), transpose_b=True) \
                + eye
        # A [f,2,2] linear systems.

        print(A)
        print(b)

        u = tf.matmul( tf.linalg.inv(A), tf.expand_dims(b, axis=-1))
        # [f,2]

        return (VF1 - VF0) * u[:,0] + (VF2 - VF0) * u[:,1]

    c2_lap = 3

    dirac = SparseDiracOperator(mesh, VF_lap, fun_lap, c2_lap)

    return dirac


def model_linear_lap_100(mesh, k=20):

    mesh.K_succ = sn.succ(mesh.K)
    mesh.K_prev = sn.prev(mesh.K)
    mesh.FM = face_mask(mesh.F)

    [batch, n, _] = mesh.V.get_shape().as_list()

    b = tf.ones([batch, n, 1], tf.float32)

    x0 = init_guess_like(b)

    def matvec(X):
        epsilon = 0 # 1e-6
        return sn.laplacian_matvec(X, mesh) + X + epsilon * X

    return conjugate_gradient( matvec, b, x0, k=k)


def model_heat_kernel_100(mesh):

    [batch, n, _] = mesh.V.get_shape().as_list()

    b = tf.zeros([batch, n, 1], tf.float32)

    x0 = init_guess_like(b)

    def lap_matvec(X):
        epsilon = 0#1e-6
        return sn.laplacian_matvec( X, mesh) + epsilon * X

    def matvec(X):
        return expm_matvec( lap_matvec, X)

    return conjugate_gradient( matvec, b, x0, k=10)


def avg_pooling(C0, mesh0, mesh1, ds01):

    [batch, n0, _] = mesh0.V.get_shape().as_list()
    [batch, n1, _] = mesh1.V.get_shape().as_list()

    [batch, _, c] = C0.get_shape().as_list()

    idm = ds01.index.get_shape().as_list()[2] # index dimension

    bindex = tf.stack([i * tf.ones([n1], dtype=tf.int32) for i in range(batch)])


    '''
    C1 = tf.stack([
        tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, 0]], axis=2)) +
        tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, 1]], axis=2)) +
        tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, 2]], axis=2)) +
        tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, 3]], axis=2))
        for i in range(c)],axis=-1)
    '''

    C1 = tf.stack([
            tf.add_n([
                     tf.multiply(
                         ds01.weight[:,:,j],
                         tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, j]], axis=2))
                     )
            for j in range(idm)])
        for i in range(c)],
        axis=-1)

    return C1 # do not need this with weights: C1*(1.0/idm)


def max_pooling(C0, mesh0, mesh1, ds01):

    [batch, n0, _] = mesh0.V.get_shape().as_list()
    [batch, n1, _] = mesh1.V.get_shape().as_list()

    [batch, _, c] = C0.get_shape().as_list()

    idm = ds01.index.get_shape().as_list()[2] # index dimension

    bindex = tf.stack([i * tf.ones([n1], dtype=tf.int32) for i in range(batch)])

    C1 = tf.stack([
            tf.reduce_max(
                tf.stack([
                         tf.gather_nd(C0[:,:,i], tf.stack([bindex, ds01.index[:, :, j]], axis=2))
                for j in range(idm)]
                , axis=-1),
            axis=2)
        for i in range(c)],
        axis=-1)

    return C1 # do not need this with weights: C1*(1.0/idm)


def check_laplacian_dependencies(mesh):

    if not hasattr(mesh, 'K_succ'):
        mesh.K_succ = sn.succ(mesh.K)
    if not hasattr(mesh, 'K_prev'):
        mesh.K_prev = sn.prev(mesh.K)
    if not hasattr(mesh, 'FM'):
        mesh.FM = face_mask(mesh.F)


def normalized_laplacian_filter(U, mesh):

    if not hasattr(mesh, 'diracg'):
        mesh.diracg = SparseGraphDiracOperator(mesh)

    [batch, n, _] = sb.get_tensor_shape(mesh.V)

    def face_fun(V, F):
        r = sn.per_face_cot_thres(V, F)
        return r

    mesh.diracg.dirac.matvec(tf.ones([batch,n,1]))

    r = generalized_laplacian_filter(U, mesh, per_face_feature_fun=face_fun)

    return


def generalized_laplacian_filter(U, mesh, per_face_feature_fun=sn.per_face_cot):

    if not hasattr(mesh, 'K_succ'):
        mesh.K_succ = sn.succ(mesh.K)
    if not hasattr(mesh, 'K_prev'):
        mesh.K_prev = sn.prev(mesh.K)

    # This is n.s.d.
    lap_op0 = SparseLaplacianOperator(mesh, per_face_feature_fun=per_face_feature_fun)

    return -lap_op0.apply_to(U)


def generalized_laplacian_filter_dense(U, mesh, per_face_feature_fun=sn.per_face_cot):

    check_laplacian_dependencies(mesh)

    # This is n.s.d.
    lap_op0 = LaplacianOperator(mesh, per_face_feature_fun=per_face_feature_fun)

    return -lap_op0.apply_to(U)


def generalized_laplacian_filter_old(U, mesh, per_face_feature_fun=sn.per_face_cot):

    check_laplacian_dependencies(mesh)
    # U
    # [batch, n, uc]

    # L0 = sn.generalized_half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_succ, mesh.FM, per_face_feature_fun)
    L0 = sn.half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_succ, mesh.FM, per_face_feature_fun)
    # [batch, n, k]
    V0 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_prev)
    # [batch, n, k, uc]

    # L1 = sn.generalized_half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_prev, mesh.FM, per_face_feature_fun)
    L1 = sn.half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_prev, mesh.FM, per_face_feature_fun)
    # [batch, n, k]
    V1 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_succ)
    # [batch, n, k, uc]

    _, _, k, uc = V0.get_shape().as_list()

    for i in range(k):
        dW = (tf.multiply(tf.expand_dims(L0[:, :, i], -1), V0[:, :, i, :] - U)
              + tf.multiply(tf.expand_dims(L1[:, :, i], -1), V1[:, :, i, :] - U)) / 2
        if i == 0:
            W = dW
        else:
            W = W + dW
    return W


def LumpMassDiagonalOperator(mesh):
    # The old implementation: mesh.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.Fm, mesh.J))
    # Now get rid of the use of J.
    return DiagonalOperator((SparseFEMAdjacency(mesh, transpose=True).apply_to(tf.expand_dims(mesh.DA / 6.0, axis=-1)))[:,:,0])

#class DiracOperator():

#    def __init__(self, V, F, J, K_succ, K_prev, FM):
#        self.D =


class DiagonalOperator():
    def __init__(self, D):
        # assert(D.shape== [batch,n]
        self.d = D

    def apply_to(self, U):
        return batch_diagmul(self.d, U)

    def matvec(self, U):
        return batch_diagmul(self.d, U)

    def invmatvec(self, U, epsilon=1e-7, epsilonB=1e-7):
        return sb.batch_diaginv(self.d, U, epsilon=epsilon, epsilonB=epsilonB) # + U * 1e-7


class LaplacianOperator():
    # This operator is negative semi-definite.
    def __init__(self, mesh, per_face_feature_fun=sn.per_face_cot):
        check_laplacian_dependencies(mesh)

        [batch, f, _] = mesh.F.get_shape().as_list()
        [_, n, k] = mesh.J.get_shape().as_list()

        mesh.bindex = tf.stack([i * tf.ones([n], dtype=mesh.J.dtype) for i in range(batch)])

        self.L0 = HalfLaplacianOperator(mesh.V, mesh.F, mesh.J, mesh.K_succ, mesh.K_prev, mesh.FM, mesh.bindex, per_face_feature_fun)
        self.L1 = HalfLaplacianOperator(mesh.V, mesh.F, mesh.J, mesh.K_prev, mesh.K_succ, mesh.FM, mesh.bindex, per_face_feature_fun)

    def apply_to(self, U):

        return self.L0.apply_to(U) + self.L1.apply_to(U)


class LaplacianSpectralPreconditioner():
    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    # P=M^-1, M is approximating A, P approximating the inverse of A.
    # This operator is positive semi-definite.
    # gamma is the coefficient for the identity regularization.
    def __init__(self, VV, DD, skip_first=True):
        DD = tf.abs(DD)
        self.gamma = tf.reciprocal(DD[:,-1]) #  # tf.ones_like(DD[:, 1])# could be too large. # DD[:, 1] # first nonzero eigenvalues.
        if skip_first:
            self.VV = VV[:,:,1:]
            self.DD = DD[:,1:]
        else:
            self.VV = VV
            self.DD = DD
        # print('LaplacianSpectralPreconditioner: sizes()')
        # print(self.VV.shape) # (2, 512, 39)
        # print(self.DD.shape) # (2, 39)
    def apply_to(self, U):
        # print('LaplacianSpectralPreconditioner: sizes()')
        # print(self.gamma.shape) # (2, )
        # print(U.shape) # (2, 512, 1)
        '''
        T0 = batch_trans(self.VV)
        print(T0.shape) # (2, 39, 512)
        T1 = batch_matmul(T0, U)
        print(T1.shape) # (2, 39, 1)
        T2 = batch_diagmul(self.DD, T1)
        print(T2.shape)
        T3 = batch_matmul(self.VV, T2)
        '''

        U2 = tf.multiply( tf.expand_dims(tf.expand_dims(self.gamma,-1),-1) , U) + batch_matmul(self.VV, batch_diagmul( tf.reciprocal(self.DD)-tf.expand_dims(self.gamma,axis=-1), batch_matmul(batch_trans(self.VV),U)))

        '''
        print('LaplacianSpectralPreconditioner:')

        print('U shape:')
        print(U.get_shape())

        print('U_t1 shape:')
        # print((tf.multiply( self.gamma, U)).get_shape()) # this is incorrect!
        print((tf.multiply(tf.expand_dims(tf.expand_dims(self.gamma, -1), -1), U).get_shape()))

        print('U_t2 shape:')
        print((batch_matmul(self.VV, batch_diagmul( tf.reciprocal(self.DD)-tf.expand_dims(self.gamma,axis=-1), batch_matmul(batch_trans(self.VV),U)))).get_shape())

        print('U2 shape:')
        print(U2.get_shape())
        '''

        # return U
        # return U2 / self.gamma
        return U2




class SparseGraphDiracOperator():

    def __init__(self, mesh):

        [batch, n, _] = sd.get_tensor_shape(mesh.V)

        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n - 1)

        self.dirac = sd.DiracOperator(sd.batch_wise_sparse_dirac(n, mesh.Fm, [], 'ones', 1, scatter_dims=True, transpose=True, aug_face=True), tranpose=True, aug_face=True )

        self.dense = None
        self.squared_dense = None

    def to_dense(self):
        if self.dense is None:
            self.dense = self.dirac.to_dense()
        return self.dense


class SparseDiracOperator():

    def __init__(self, mesh, VF, fun, c2):

        assert(hasattr(mesh, 'per_face_doublearea'))

        [batch, n, c1] = sd.get_tensor_shape(VF)

        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n - 1)

        self.dirac = sd.DiracOperator( sd.batch_wise_sparse_dirac(n, mesh.Fm, VF, fun, c2, scatter_dims=True, aug_face=True), aug_face=True )

        # self.ops = [[ for i in range(c2)] for i in range(batch)]

        self.dense = None
        self.squared_dense = None

        self.DA = mesh.per_face_doublearea

    def to_dense(self):
        if self.dense is None:
            self.dense = self.dirac.to_dense()
        return self.dense

    def to_squared_dense(self):

        if self.squared_dense is None:
            weights = tf.tile(tf.expand_dims(self.DA,-1), multiples=[1,1,3])
            print('to_squared_dense:',weights)
            self.squared_dense = self.dirac.to_squared_dense(weights=weights)
        return self.squared_dense


#################################################
# FEM Version 1
#################################################


class SparseFEMOperatorBase_v0():

    def __init__(self, mesh):
        [batch, n, _] = sd.get_tensor_shape(mesh.V)
        self.n = n
        self.batch = batch

        # this is a tmp hack, recover: removed: mesh.F = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n-1)
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = None

    # self.ops must be set before any further computing happens.

    def apply_to(self, U):
        #[batch, _, _] = sd.get_tensor_shape(U)
        batch = self.batch
        return tf.stack([tf.sparse_tensor_dense_matmul(sp_a=self.ops[bi], b=U[bi,:,:]) for bi in range(batch)], axis=0)

    def to_dense(self):
        batch = self.batch
        if not hasattr(self, 'dense'):
            self.dense = tf.stack([tf.sparse_tensor_to_dense(self.ops[bi]) for bi in range(batch)], axis=0) # , validate_indices=False
        return self.dense


class SparseFEMMass_v0(SparseFEMOperatorBase_v0):

    def __init__(self, mesh):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        # old implementation.
        # self.ops = sd.batch_wise_sparse_fem_old(self.n, mesh.Fm, mesh.DA, VC=tf.zeros([self.batch, self.n, 1]),
        #                                        assembler=sd.mass_assembler_old)  # remember to change here back too!

        self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=tf.zeros([self.batch,self.n,1]),
                                                assembler=sd.mass_assembler_v0)


class SparseFEMLap_v0(SparseFEMOperatorBase_v0):

    def __init__(self, mesh):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=mesh.V,
                                                assembler=sd.lap_assembler_v0)


# This implements a general p.s.d. operator.
# It allows learnable parameters through assembler.
class SparseFEMGeneral_v0(SparseFEMOperatorBase_v0):

    def __init__(self, mesh, VC, assembler):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=VC,
                                                assembler=assembler)



#################################################
# FEM Version 2
#################################################

class SparseFEMOperatorBase():

    def __init__(self, mesh, use_old_tensor_rep=False):
        [batch, n, _] = sd.get_tensor_shape(mesh.V)
        self.n = n
        self.batch = batch

        self.use_old_tensor_rep = use_old_tensor_rep

        # this is a tmp hack, recover: removed: mesh.F = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n-1)
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = None

    # self.ops must be set before any further computing happens.
    # in case cdim of op is larger than one, dim of U will be multipled by cdim.
    def apply_to(self, U):
        # U: [batch, n, udim] or [batch, f, udim]
        return sd.sparse_tensors_apply(self.ops, U, self.use_old_tensor_rep, keep_dim=False)

    def apply_to2(self, U):
        # U: [batch, n, udim] or [batch, f, udim]
        # output is [batch,n/f, udim, cdim]
        return sd.sparse_tensors_apply(self.ops, U, self.use_old_tensor_rep, keep_dim=True)

    def trans_apply_to(self, U):
        # U: [batch, n, udim] or [batch, f, udim]
        return sd.sparse_tensors_trans_apply(self.ops, U)

    def to_dense(self):
        # [batch, ...,..., cdim]
        if not hasattr(self, 'dense'):
            # old: self.dense = sd.sparse_tensors_convert_to_dense(self.ops, self.use_old_tensor_rep)
            # the new one for fix pattern operators
            self.dense = tf.stack([self.ops[i].to_dense() for i in range(len(self.ops))], axis=0)
        return self.dense


# This is the squared root of Laplacian, but it is not the Dirac operator.
# Gradients of hat basis functions.
class SparseFEMGradient(SparseFEMOperatorBase):

    def __init__(self, mesh):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        VC = mesh.V
        import surf_op as so
        V2AFC = so.operator_vertex_to_face_aug(VC, mesh.Fm)

        slicer = sd.FEMSlicer(VC=None, DEC=V2AFC, V2AFC=V2AFC)
        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                                assembler=sd.grad_assembler_pos, transpose=False, use_old_tensor_rep=self.use_old_tensor_rep)
        return


class SparseDi():
# 4f by 4n
    def __init__(self, mesh, transpose=False):
        if not hasattr(mesh, 'SparseDi'):
            self.Di = SparseFEMDirac(mesh, strong=True, transpose=transpose)
            mesh.SparseDi = self.Di
        else:
            self.Di = mesh.SparseDi
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        [batch, f, _] = sb.get_tensor_shape(mesh.F)
    def apply_to(self, U):

        [batch, n4, udim] = sb.get_tensor_shape(U)
        U2 = tf.reshape(U, shape=[batch, n4 // 4, 4, udim])

        # a==0
        # b: +[1,0] -[0,1] +[3,2] -[2,3]
        # c: +[2,0] -[0,2] +[1,3] -[3,1]
        # d: +[3,0] -[0,3] +[2,1] -[1,2]

        # [0,1]

        # [batch, f, udim, 0,1,2] # 0,1,2 -> b,c,d
        QT0 = self.Di.apply_to2(U2[:, :, 0, :])
        QT1 = self.Di.apply_to2(U2[:, :, 1, :])
        QT2 = self.Di.apply_to2(U2[:, :, 2, :])
        QT3 = self.Di.apply_to2(U2[:, :, 3, :])
        #

        a = -1 # not needed
        b = 0
        c = 1
        d = 2
        M = [[] in range(4)]

        '''
        M[0] =  +QT0[a] -QT1[b] -QT2[c] -QT3[d]
        M[1] =  +QT0[b] +QT1[a] -QT2[d] +QT3[c]
        M[2] =  +QT0[c] +QT1[d] +QT2[a] -QT3[b]
        M[3] =  +QT0[d] -QT1[c] +QT2[b] +QT3[a]
        '''
        M[0] =          -QT1[:, :, b] -QT2[:, :, c] -QT3[:, :, d]
        M[1] =   QT0[:, :, b]         -QT2[:, :, d] +QT3[:, :, c]
        M[2] =   QT0[:, :, c] +QT1[:, :, d]         -QT3[:, :, b]
        M[3] =   QT0[:, :, d] -QT1[:, :, c] +QT2[:, :, b]
        # M[i]: [batch, f, udim]

        # [batch, 4f, udim]
        return tf.concat(M, axis=1)


class SparseDiA():
# 4n by 4f
    def __init__(self, mesh):
        if not hasattr(mesh, 'Di'):
            self.Di = SparseFEMDirac(mesh, strong=True, transpose=False)
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        [batch, f, _] = sb.get_tensor_shape(mesh.F)
    def apply_to(self, U):

        [batch, n4, udim] = sb.get_tensor_shape(U)
        U2 = tf.reshape(U, shape=[batch, n4 // 4, 4, udim])

        # a==0
        # b: +[1,0] -[0,1] +[3,2] -[2,3]
        # c: +[2,0] -[0,2] +[1,3] -[3,1]
        # d: +[3,0] -[0,3] +[2,1] -[1,2]

        # [0,1]

        # [batch, f, udim, 0,1,2] # 0,1,2 -> b,c,d
        QT0 = self.Di.apply_to2(U2[:, :, 0, :])
        QT1 = self.Di.apply_to2(U2[:, :, 1, :])
        QT2 = self.Di.apply_to2(U2[:, :, 2, :])
        QT3 = self.Di.apply_to2(U2[:, :, 3, :])
        #

        a = -1 # not needed
        b = 0
        c = 1
        d = 2
        M = [[] in range(4)]

        '''
        M[0] =  +QT0[a] -QT1[b] -QT2[c] -QT3[d]
        M[1] =  +QT0[b] +QT1[a] -QT2[d] +QT3[c]
        M[2] =  +QT0[c] +QT1[d] +QT2[a] -QT3[b]
        M[3] =  +QT0[d] -QT1[c] +QT2[b] +QT3[a]
        '''
        M[0] =          -QT1[b] -QT2[c] -QT3[d]
        M[1] =  -QT0[b]         -QT2[d] +QT3[c]
        M[2] =  -QT0[c] +QT1[d]         -QT3[b]
        M[3] =  +QT0[d] -QT1[c] +QT2[b]
        # M[i]: [batch, f, udim]

        # [batch, 4f, udim]
        return tf.concat(M, axis=1)


class SparseDiA():
# 2n by 2f
    def __init__(self, mesh):
        if not hasattr(mesh, 'DiA'):
            self.DiA = SparseFEMDirac(mesh, strong=False, transpose=True)
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        [batch, f, _] = sb.get_tensor_shape(mesh.F)


class SparseFEMDirac(SparseFEMOperatorBase):

    def __init__(self, mesh, strong=False, transpose=False):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        if not hasattr(mesh, 'DE'):
            mesh.DE = sd.per_face_dir_edges(mesh.V, mesh.Fm)

        slicer = sd.FEMSlicer(VC=None, DEC=mesh.DE, V2AFC=None)
        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                            assembler=sd.dirac_strong_assembler if strong else sd.dirac_weak_assembler, transpose=transpose,
                                            use_old_tensor_rep=self.use_old_tensor_rep)
        return


class SparseFEMWedge(SparseFEMOperatorBase):

    def __init__(self, mesh, transpose=False):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        if not hasattr(mesh, 'DE'):
            mesh.DE = sd.per_face_dir_edges(mesh.V, mesh.Fm)

        slicer = sd.FEMSlicer(VC=None, DEC=mesh.DE, V2AFC=None)
        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                            assembler=sd.wedge_assembler,
                                            transpose=transpose,
                                            use_old_tensor_rep=self.use_old_tensor_rep)
        return


# This yields [batch][f,n][1] face-vertex adjacency matrix.
class SparseFEMAdjacency(SparseFEMOperatorBase):

    def __init__(self, mesh, transpose=False):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        [batch,f,_] = sb.get_tensor_shape(mesh.F)
        slicer = sd.FEMSlicer(VC=None, DEC=None, V2AFC=tf.zeros([batch, f, 3, 1]))
        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                            assembler=sd.grad_assembler_adjacency, transpose=transpose,
                                            use_old_tensor_rep=self.use_old_tensor_rep)
        return


# This is the squared root of Full Mass Matrix.
class SparseFEMFV(SparseFEMOperatorBase):

    def __init__(self, mesh):
        super().__init__(mesh)

        assert False # still working on it.

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        VC = mesh.V
        import surf_op as so
        V2AFC = so.operator_vertex_to_face_aug(VC, mesh.Fm)

        slicer = sd.FEMSlicer(VC=None, DEC=V2AFC, V2AFC=V2AFC)
        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                            assembler=sd.grad_assembler_pos, transpose=False,
                                            use_old_tensor_rep=self.use_old_tensor_rep)
        return


# This implements a general assymetric operator.
# It allows learnable parameters through assembler.
# face by vertex by cdim
class SparseFEMGeneral(SparseFEMOperatorBase):

    def __init__(self, mesh, slicer, assembler, transpose=False):
        super().__init__(mesh)

        # example to setup:
        # slicer = sd.FEMSlicer(VC=VC, DEC=DEC, V2AFC=V2AFC)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = sd.batch_wise_sparse_fem(self.n, mesh.Fm, slicer=slicer,
                                               assembler=assembler, transpose=transpose, use_old_tensor_rep=self.use_old_tensor_rep)


class SparseFEMBiBase():

    def __init__(self, mesh, femop):

        # assume op instantiate a subclass of SparseFEMOperatorBase

        self.femop = femop
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))

        self.batch = len(self.femop.ops)
        self.DA = mesh.DA

        for bi in range(len(self.femop.ops)):
            sd.create_transposed_operator(self.femop.ops[bi])

        return

    def apply_to(self, U):

        # U: [batch,n,udim]
        # femop are [batch] operators, each of dimension [n,m,cdim]
        # see: sd.sparse_tensors_apply()
        import surf_basic as sb
        batch, _, udim = sb.get_tensor_shape(U)

        cdim = self.femop.ops[0].cdim

        assert batch==self.batch

        # cc = 0
        # return tf.stack([tf.sparse_tensor_dense_matmul(sp_a=self.ops[bi][cc], b=U[bi,:,:]) for bi in range(batch)], axis=0)

        # #tf.stack(
        return tf.stack([
                    tf.linalg.trace(tf.reshape(
                        self.femop.ops[bi].transpose.apply_to2(
                            tf.multiply(tf.expand_dims(self.DA[bi,:]*0.5, axis=-1), # DA: [f,] --> [f,1]
                                tf.reshape(self.femop.ops[bi].apply_to2(U[bi, :, :]), shape=[-1, udim*cdim]) #[m,udim,cdim]=[f,udim,3] --> [f,udim*3]
                            ) # mul -> [f, udim*3]
                        ), # [f,udim*3,3]
                    shape=[-1,udim,cdim,cdim])) # --> [f,udim,3,3] --> [f,udim]: trace sums up along the last two dims.
            for bi in range(batch)], axis=0)

    def to_dense(self):
        return self.apply_to(tf.stack([tf.eye(self.femop.n) for i in range(self.batch)], axis=0))


class SparseFEMBiGradient(SparseFEMBiBase):
    # A.k.a. Laplacian, as bi-gradient operator.
    def __init__(self, mesh):
        super().__init__(mesh, SparseFEMGradient(mesh))


#################################################
# FEM Galerkin
#################################################


class SparseFEMGalerkinBase(SparseFEMOperatorBase):

    def __init__(self, mesh):
        super().__init__(mesh, use_old_tensor_rep=False)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))


# This implements a general p.s.d. operator.
# It allows learnable parameters through assembler.
class SparseFEMGalerkinGeneral(SparseFEMOperatorBase):

    def __init__(self, mesh, VC, assembler):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=VC,
                                               assembler=assembler)


class SparseFEMGalerkinMass(SparseFEMOperatorBase):

    def __init__(self, mesh):
        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        # old implementation.
        # self.ops = sd.batch_wise_sparse_fem_old(self.n, mesh.Fm, mesh.DA, VC=tf.zeros([self.batch, self.n, 1]),
        #                                        assembler=sd.mass_assembler_old)  # remember to change here back too!

        if True:
            slicer = sd.mass_slicer_galerkin(mesh.V, mesh.Fm)
            self.ops = sd.batch_wise_sparse_fem_galerkin(self.n, mesh.Fm, mesh.DA, slicer=slicer,
                                                   assembler_galerkin=sd.mass_assembler_galerkin)
        #else: # This old code no longer works
        #    self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=tf.zeros([self.batch, self.n, 1]),
        #                                       assembler=sd.mass_assembler_v0)


class SparseFEMGalerkinLap(SparseFEMOperatorBase):

    def __init__(self, mesh, assembler=sd.lap_assembler_galerkin_robust):

        # sd.lap_assembler_galerkin: exact laplacian assembler, vulnerable to noises.
        # sd.lap_assembler_galerkin_robust robust but approximate

        super().__init__(mesh)

        # Specific to this FEM operator
        if not hasattr(mesh, 'DA'):
            setattr(mesh, 'DA', sd.per_face_doublearea(mesh.V, mesh.F))
        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        if True:
            slicer = sd.lap_slicer_galerkin(mesh.V, mesh.Fm)
            self.ops = sd.batch_wise_sparse_fem_galerkin(self.n, mesh.Fm, mesh.DA, slicer=slicer,
                                                   assembler_galerkin=assembler)
        #else: # This old code no longer works
        #    self.ops = sd.batch_wise_sparse_fem_v0(self.n, mesh.Fm, mesh.DA, VC=mesh.V,
        #                                       assembler=sd.lap_assembler_v0)


#################################################


class SparseLaplacianOperator():

    def __init__(self, mesh, per_face_feature_fun=sn.per_face_cot):
        [batch, n, _] = sd.get_tensor_shape(mesh.V)
        self.batch = batch

        # Specific to Laplacian
        self.n = n

        if callable(per_face_feature_fun):
            FC = per_face_feature_fun(mesh.V, mesh.F)
        else:
            FC = per_face_feature_fun

        if not hasattr(mesh, 'Fm'):
            mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=self.n - 1)

        self.ops = sd.batch_wise_sparse_laplacian(n, mesh.Fm, FC)

    def apply_to(self, U):
        #[batch, _, _] = sd.get_tensor_shape(U)
        batch = self.batch
        return tf.stack([tf.sparse_tensor_dense_matmul(sp_a=self.ops[bi], b=U[bi,:,:]) for bi in range(batch)], axis=0)

    def to_dense(self):
        batch = self.batch
        if not hasattr(self, 'dense'):
            self.dense = tf.stack([tf.sparse_tensor_to_dense(self.ops[bi]) for bi in range(batch)], axis=0) # , validate_indices=False
        return self.dense


def sparse_laplacian(self, mesh, per_face_feature_fun):
    if callable(per_face_feature_fun):
        FC = per_face_feature_fun(mesh.V, mesh.F)
    else:
        FC = per_face_feature_fun
    [batch, n, _] = sd.get_tensor_shape(mesh.V)
    self.op = sd.sparse_laplacian(n, mesh.F, mesh.J, FC)


class HalfLaplacianOperator():

    def __init__ (self, V, F, J, K_succ, K_prev, FM, bindex, per_face_feature_fun):
        self.weights = sn.half_laplacian(V, F, J, K_succ, FM, per_face_feature_fun=per_face_feature_fun)
        self.indices = sn.half_laplacian_index(F, J, K_prev, bindex)
        self.bindex = bindex

    def apply_to(self, U):

        V0 = sn.half_laplacian_apply(U, self.indices, self.weights, self.bindex)

        # print('HalfLaplacianOperator:apply_to()')
        # print(V0.get_shape())

        return V0

    def apply_to_old(self, U):

        V0 = sn.half_laplacian_expand_apply(U, self.indices)

        _, _, k, uc = V0.get_shape().as_list()

        for i in range(k):
            dW = (tf.multiply(tf.expand_dims(self.weights[:, :, i], -1), V0[:, :, i, :] - U)) / 2
            if i == 0:
                W = dW
            else:
                W = W + dW
        return W


# Not Finished.
class MassFullOperator():
    def __init__(self, V, F, J, K, FM):
        return

    def apply_to(self, U):
        return


def laplacian_filter_old(U, mesh):

    check_laplacian_dependencies(mesh)

    # U
    # [batch, n, uc]

    L0 = sn.half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_succ, mesh.FM)
    # [batch, n, k]
    V0 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_prev)
    # [batch, n, k]

    L1 = sn.half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_prev, mesh.FM)
    # [batch, n, k]
    V1 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_succ)
    # [batch, n, k]

    _, _, k, uc = V0.get_shape().as_list()

    for i in range(k):
        dW = (tf.multiply(tf.expand_dims(L0[:, :, i], -1), V0[:, :, i, :] - U)
              + tf.multiply(tf.expand_dims(L1[:, :, i], -1), V1[:, :, i, :] - U)) / 2
        if i == 0:
            W = dW
        else:
            W = W + dW
    return W


def laplacian_filter(U, mesh):

    check_laplacian_dependencies(mesh)

    # U
    # [batch, n, uc]

    # old lap_op = LaplacianOperator(mesh.V, mesh.F, mesh.J, K_succ, K_prev, mesh.FM)
    lap_op = LaplacianOperator(mesh)

    return lap_op.apply_to(U)


def modified_laplacian_filter(U, mesh, E):

    check_laplacian_dependencies(mesh)

    # U
    # [batch, n, uc]

    L0 = sn.half_laplacian_from_edge(mesh.V, mesh.F, mesh.J, mesh.K_succ, mesh.FM, E)
    # [batch, n, k]
    V0 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_prev)
    # [batch, n, k]

    L1 = sn.half_laplacian(mesh.V, mesh.F, mesh.J, mesh.K_prev, mesh.FM, E)
    # [batch, n, k]
    V1 = sn.half_laplacian_smoothing_pre(U, mesh.F, mesh.J, mesh.K_succ)
    # [batch, n, k]

    _, _, k, uc = V0.get_shape().as_list()

    for i in range(k):
        dW = (tf.multiply(tf.expand_dims(L0[:, :, i], -1), V0[:, :, i, :] - U)
              + tf.multiply(tf.expand_dims(L1[:, :, i], -1), V1[:, :, i, :] - U)) / 2
        if i == 0:
            W = dW
        else:
            W = W + dW
    return W


def heat_kernel_from_eigen_old(mesh, time, idx):
    # this implements VV*exp(-time*DD)*VV'*M(:,idx)

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if not hasattr(mesh, 'Mass'):
        mesh.Mass = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    #def matvec(VV):
    #    return default_batch_inner_by_index2(VV,idx)

    def matvec(VV):
        [batch, n, _] = VV.get_shape().as_list()
        delta = batch_delta(batch, n, idx)
        # u = delta
        u = mesh.Mass.matvec(delta)
        return batch_matmul(batch_trans(VV), u)

    HKS = heat_kernel_from_eigen_core( mesh.A[:, :, 50:90], mesh.b[:, 50:90], time=time, matvec=matvec, skip_first=False)

    HKS = tf.multiply(tf.expand_dims(mesh.VM, -1), HKS)

    return HKS


def heat_kernel_signature_from_eigen_old2(mesh, time):

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if not hasattr(mesh, 'Mass'):
        mesh.Mass = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))
        #mesh.Mass = LumpMassDiagonalOperator(mesh)

    HKS = heat_kernel_signature_from_eigen_core( mesh.A[:, :, 0:20], mesh.b[:, 0:20], time=time, skip_first=False)

    HKS = tf.multiply(tf.expand_dims(mesh.VM, -1), HKS)

    return HKS


def heat_kernel_signature_from_eigen(mesh, time):

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if not hasattr(mesh, 'Mass'):
        #mesh.Mass = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))
        mesh.Mass = LumpMassDiagonalOperator(mesh)

    HKS = heat_kernel_signature_from_eigen_core( mesh.A[:, :, 0:20], mesh.b[:, 0:20], time=time, skip_first=False)

    HKS = tf.multiply(tf.expand_dims(mesh.VM, -1), HKS)

    return HKS


def wave_kernel_signature_from_eigen(mesh, time):

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if not hasattr(mesh, 'Mass'):
        mesh.Mass = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    WKS = wave_kernel_signature_from_eigen_core( mesh.A[:, :, 0:20], mesh.b[:, 0:20], time=time, skip_first=False)

    WKS = tf.multiply(tf.expand_dims(mesh.VM, -1), WKS)

    return WKS


def biharmonic_embedding_from_eigen(mesh):

    return biharmonic_embedding_from_eigen_core(mesh.A[:, :, 0:20], mesh.b[:, 0:20])


def heat_kernel_from_eigen(mesh, time, idx):
    # this implements VV*exp(-time*DD)*VV'*M(:,idx)

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if not hasattr(mesh, 'Mass'):
        #mesh.Mass = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))
        mesh.Mass = LumpMassDiagonalOperator(mesh)

    #def matvec(VV):
    #    return default_batch_inner_by_index2(VV,idx)

    def matvec(VV):
        [batch, n, _] = VV.get_shape().as_list()
        delta = batch_delta(batch, n, idx)
        # u = delta
        u = mesh.Mass.matvec(delta)
        return batch_matmul(batch_trans(VV), u)

    HK = heat_kernel_from_eigen_core( mesh.A[:, :, 0:20], mesh.b[:, 0:20], time=time, matvec=matvec, skip_first=False)

    HK = tf.multiply(tf.expand_dims(mesh.VM, -1), HK)

    return HK


def heat_kernel_from_eigen2(mesh, time, idx):
    # this implements VV*exp(-time*DD)*VV'*I(:,idx)

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    def matvec(VV):
        [batch, n, _] = VV.get_shape().as_list()
        delta = batch_delta(batch, n, idx)
        u = delta
        return batch_matmul(batch_trans(VV), u)

    HK = heat_kernel_from_eigen_core( mesh.A[:, :, 0:20], mesh.b[:, 0:20], time=time, matvec=matvec, skip_first=False)

    HK = tf.multiply(tf.expand_dims(mesh.VM, -1), HK)

    return HK


def batch_delta(batch,n,idx):
    delta = tf.concat([tf.zeros([batch, idx, 1], tf.float32),
                       tf.ones([batch, 1, 1], tf.float32),
                       tf.zeros([batch, n - 1 - idx, 1], tf.float32)], axis=1)
    # [batch,n,1]
    return delta


def default_batch_inner_by_matrix(VV,u):
    # this computes VV^T u
    return batch_matmul(batch_trans(VV), u)


def default_batch_inner_by_index(VV, idx):
    print('Warning: this might have a bug, check it before use.')
    return tf.expand_dims( VV[:, idx, :], axis=-1)


# Untested: double check before use.
def default_batch_inner_by_index2(VV, idx):
    [batch, n, _] = VV.get_shape().as_list()
    delta = batch_delta(batch,n,idx)
    return batch_matmul(batch_trans(VV), delta)


def biharmonic_embedding_from_eigen_core( VV, DD, p=2,
                                 skip_first=True):
    # This yields VV * exp(-t*DD) * VV^T * u

    # The oracle VV^T * u shall be given.

    assert(p==2)

    DD = tf.abs(DD)

    if skip_first:
        VV = VV[:,:,1:]
        DD = DD[:,1:]
    else:
        VV = VV
        DD = DD

    epsilon = 1e-6

    # [batch, n, k-1] or k if not skipping.
    return batch_matmul(VV, tf.expand_dims(tf.reciprocal(DD+epsilon), axis=-1) )


def wave_kernel_signature_from_eigen_core( VV, DD, time,
                                 skip_first=False):
    # This yields VV * exp(-t*DD) * VV^T * u

    # The oracle VV^T * u shall be given.




    DD = tf.abs(DD)

    if skip_first:
        VV = VV[:,:,1:]
        DD = DD[:,1:]
    else:
        VV = VV
        DD = DD

    # [batch, n, 1]
    return batch_matmul(tf.square(VV), tf.expand_dims(tf.exp(-time*DD), axis=-1) )


def heat_kernel_signature_from_eigen_core( VV, DD, time,
                                 skip_first=False):
    # This yields VV * exp(-t*DD) * VV^T * u

    # The oracle VV^T * u shall be given.


    DD = tf.abs(DD)

    if skip_first:
        VV = VV[:,:,1:]
        DD = DD[:,1:]
    else:
        VV = VV
        DD = DD

    # [batch, n, 1]
    return batch_matmul(tf.square(VV), tf.expand_dims(tf.exp(-time*DD), axis=-1) )


def heat_kernel_from_eigen_core( VV, DD, time,
                                 matvec=lambda vv:default_batch_inner_by_index(vv,0),
                                 skip_first=False):
    # This yields VV * exp(-t*DD) * VV^T * u

    # The oracle VV^T * u shall be given.


    DD = tf.abs(DD)

    if skip_first:
        VV = VV[:,:,1:]
        DD = DD[:,1:]
    else:
        VV = VV
        DD = DD

    T1 = matvec(VV)
    # T1: (2, 39, 1)
    HKS = batch_matmul( VV, batch_diagmul(tf.exp(-time*DD),  T1 ))

    # [batch, n, 1]
    return HKS


def heat_kernel_approx_pre(mesh, b=None, x0=None):

    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if b is None:
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        # b = tf.zeros([batch, n, 1], tf.float32)
        b = tf.concat([tf.ones([batch, 1, 1], tf.float32), tf.zeros([batch, n - 1, 1], tf.float32)], axis=1)
    elif b == 'ones':
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        b = tf.ones([batch, n, 1], tf.float32)
    elif b == 'full':
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        b = tf.tile(tf.expand_dims(tf.eye(n), axis=0), [batch,1,1]) #tf.eye(n, batch_shape=batch, dtype=tf.float32)
    else:
        assert type(b) is int
        [batch, n, _] = sb.get_tensor_shape(mesh.V)
        assert b < n
        b = batch_delta(batch, n, idx=b)

    b = tf.multiply(tf.expand_dims(mesh.VM, -1), b)  # VM: [batch,n], b: [batch, n, c]

    if x0 == None:
        x0 = init_guess_like(b)
    x0 = tf.multiply(tf.expand_dims(mesh.VM, -1), x0)

    return b, x0


def generalized_heat_kernel_signature_approx(mesh, k, time, x0=None, pre_cond='spectral', per_face_feature_fun=sd.per_face_cot):

    # hk = generalized_heat_kernel_approx(mesh, k, time, b='full', x0=x0, per_face_feature_fun=per_face_feature_fun)

    hk = heat_kernel_approx2(mesh, k, time, b='full', x0=x0, pre_cond=pre_cond)

    # get the diagonal of
    return tf.expand_dims(tf.linalg.diag_part(hk), axis=-1)


def generalized_heat_kernel_approx(mesh, k, time, b=None, x0=None, per_face_feature_fun=sd.per_face_cot):

    b, x0 = heat_kernel_approx_pre(mesh, b=b, x0=x0)

    # This is n.s.d.

    if not hasattr(mesh, 'K_succ'):
        mesh.K_succ = sn.succ(mesh.K)
    if not hasattr(mesh, 'K_prev'):
        mesh.K_prev = sn.prev(mesh.K)

    # This is n.s.d.
    lap_op0 = SparseLaplacianOperator(mesh, per_face_feature_fun=per_face_feature_fun)
    mass_op0 = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))


    def pade(X, beta=0):
        # assert beta is non-negative.
        return -lap_op0.apply_to(X) * beta + mass_op0.apply_to(X)

    def Amatvec(X):

        epsilon = 0 # 1e-6
        # time = 0.001
        return -lap_op0.apply_to(X) * time + mass_op0.apply_to(X) + epsilon * X

    def Mmatvec(X):
        return mass_op0.apply_to(X)

    r = conjugate_gradient(Amatvec, Mmatvec(b), x0, k=k, Imatvec=Mmatvec) # but this one is better.

    r = tf.multiply(tf.expand_dims(mesh.VM,-1), r) # VM: [batch,n], b: [batch, n, c]

    return r


def heat_kernel_approx2(mesh, k, time, b=None, x0=None, pre_cond='spectral'):
    # this computes (M+time*L)^(-1)M*b

    b, x0 = heat_kernel_approx_pre(mesh, b=b, x0=x0)

    if False:
        # old implementation.
        # This is n.s.d.
        check_laplacian_dependencies(mesh)
        lap_op0 = SparseLaplacianOperator(mesh)
        mass_op0 = SparseFEMMass_v0(mesh)
        sign = -1
    else:
        lap_op0 = SparseFEMGalerkinLap(mesh,assembler=sd.lap_assembler_galerkin)
        # mass_op0 = SparseFEMGalerkinMass(mesh)
        mass_op0 = LumpMassDiagonalOperator(mesh)
        sign = 1

    # lap_op0 = LaplacianOperator(mesh)
    # mass_op0 = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    if pre_cond == 'spectral':

        if False:
            pre_op0_weak = LaplacianSpectralPreconditioner(mesh.A[:,:,0:40], mesh.b[:,0:40], skip_first=True)
            def Pmatvec_w(X):
                return pre_op0_weak.apply_to(X)
        if True:
            pre_op0_strong = LaplacianSpectralPreconditioner(mesh.A[:, :, 0:20], mesh.b[:, 0:20], skip_first=True)
            def Pmatvec_s(X):
                return pre_op0_strong.apply_to(X)

    else:
        assert pre_cond is None

    def Mmatvec(X):
        epsilon = 1e-8
        return mass_op0.apply_to(X) + epsilon * X

    '''
    if False:
        # Pade 0/1
        # 1 / (1+x)

        def Amatvec_old(X):
            epsilon = 0  # 1e-6
            # time = 0.1
            return lap_op0.apply_to(X) * (time * sign) + X + epsilon * X

        def pade(X, beta=0):
            # assert beta is non-negative.
            return lap_op0.apply_to(X) * (beta * sign) + mass_op0.apply_to(X)

        def Amatvec(X):
            epsilon = 1e-8  #
            # time = 0.001
            return lap_op0.apply_to(X) * (time * sign) + mass_op0.apply_to(X) + epsilon * X

        def Bmatvec(X):
            epsilon = 1e-8
            return mass_op0.apply_to(X) + epsilon * X

    elif False:
        # Pade 2/2 of e^(-x)

        # (12+6x+x^2)\(12-6x+x^2)

        def Amatvec(X):
            epsilon = 1e-8  #
            # time = 0.001
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) + mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X

        def Bmatvec(X):
            epsilon = 1e-8
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) - mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X

    '''

    if True:
        # Pade 3/3

        # (120+60x+12x^2+x^3)\(120-60x+12x^2-x^3)
        '''
        def Amatvec(X):
            epsilon = 1e-8  #
            # time = 0.001
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) + mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X

        def Bmatvec(X):
            epsilon = 1e-8
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) - mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X
        '''

        def P(X):
            return lap_op0.apply_to(X) * (time * sign)

        def I(X):
            return mass_op0.apply_to(X)

        def Amatvec_0_0(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) + I(X) + epsilon * X

        def Bmatvec_0_0(X):
            # epsilon = 1e-8
            return I(X) #+ epsilon * X

        # Pade 1/1
        # (1+0.5x)\(1-0.5x)
        def Amatvec_1_1(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) * 0.5 + I(X) + epsilon * X

        def Bmatvec_1_1(X):
            #epsilon = 1e-8
            return P(X) * (-0.5) + I(X) #+ epsilon * X # not necessary for B.

        Amatvec = Amatvec_1_1
        Bmatvec = Bmatvec_1_1

        Amatvec = Amatvec_0_0
        Bmatvec = Bmatvec_0_0


    #print('b shape:')
    #print(b.get_shape())

    #print('x0 shape:')
    #print(x0.get_shape())

    def dd(t):
        return tf.cast(t, tf.float64)
    def ss(t):
        return tf.cast(t, tf.float32)

    if pre_cond is None:
        # r = conjugate_gradient( Amatvec_old, b, x0, k=k)
        # r = conjugate_gradient( Amatvec, Bmatvec(b), x0, k=k) # this is also a correct use, missing mass mat.
        # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=Mmatvec) # but this one is better.
        # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=None)

        rhs = ss(tf.linalg.solve(matrix=dd(Mmatvec(sd.batch_eye_like(b))), rhs=dd(b)))
        # rhs = b
        r = tf.linalg.solve(matrix=dd(Amatvec(sd.batch_eye_like(rhs))), rhs=dd(Bmatvec(rhs)))
    else:
        #r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_w, Bmatvec(b), x0, k=k)
        r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_s, Bmatvec(b), x0, k=k, Imatvec=Mmatvec)

    r = tf.multiply(dd(tf.expand_dims(mesh.VM,-1)), r) # VM: [batch,n], b: [batch, n, c]

    #print('r shape:')
    #print(r.get_shape())

    return r


def heat_kernel_approx3(mesh, k, time, b=None, x0=None, pre_cond='spectral'):
    # this computes (M+time*L)^(-1)M*b

    b, x0 = heat_kernel_approx_pre(mesh, b=b, x0=x0)

    if False:
        # old implementation.
        # This is n.s.d.
        check_laplacian_dependencies(mesh)
        lap_op0 = SparseLaplacianOperator(mesh)
        mass_op0 = SparseFEMMass_v0(mesh)
        sign = -1
    else:
        lap_op0 = SparseFEMGalerkinLap(mesh,assembler=sd.lap_assembler_galerkin)
        # mass_op0 = SparseFEMGalerkinMass(mesh)
        mass_op0 = LumpMassDiagonalOperator(mesh)
        sign = 1

    # lap_op0 = LaplacianOperator(mesh)
    # mass_op0 = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    if pre_cond == 'spectral':

        if False:
            pre_op0_weak = LaplacianSpectralPreconditioner(mesh.A[:,:,0:40], mesh.b[:,0:40], skip_first=True)
            def Pmatvec_w(X):
                return pre_op0_weak.apply_to(X)
        if True:
            pre_op0_strong = LaplacianSpectralPreconditioner(mesh.A[:, :, 0:20], mesh.b[:, 0:20], skip_first=True)
            def Pmatvec_s(X):
                return pre_op0_strong.apply_to(X)

    else:
        assert pre_cond is None

    def Mmatvec(X):
        epsilon = 0 # 1e-8
        return mass_op0.apply_to(X) + epsilon * X


    if True:
        # Pade 3/3

        # (120+60x+12x^2+x^3)\(120-60x+12x^2-x^3)
        '''
        def Amatvec(X):
            epsilon = 1e-8  #
            # time = 0.001
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) + mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X

        def Bmatvec(X):
            epsilon = 1e-8
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) - mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X
        '''

        def P(X):
            return lap_op0.apply_to(X) * (time * sign)

        def I(X):
            return mass_op0.apply_to(X)

        def Amatvec_0_0(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) + I(X) + epsilon * X

        def Bmatvec_0_0(X):
            epsilon = 1e-8
            return I(X) + epsilon * X

        # Pade 1/1
        # (1+0.5x)\(1-0.5x)
        def Amatvec_1_1(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) * 0.5 + I(X) + epsilon * X

        def Bmatvec_1_1(X):
            #epsilon = 1e-8
            return P(X) * (-0.5) + I(X) #+ epsilon * X # not necessary for B.

        Amatvec = Amatvec_1_1
        Bmatvec = Bmatvec_1_1


    #print('b shape:')
    #print(b.get_shape())

    #print('x0 shape:')
    #print(x0.get_shape())

    def dd(t):
        return tf.cast(t, tf.float64)
    def ss(t):
        return tf.cast(t, tf.float32)

    def r2c(T):
        real = tf.cast(T, tf.float64)
        imag = tf.zeros_like(real)
        return tf.complex(real, imag)

    def c2r(T):
        return tf.cast(tf.real(T), tf.float32)


    if False:
        coeffs_real = [1]
        coeffs_imag = [0]
        poles_real = [-1]
        poles_imag = [0]
    elif False:
        coeffs_real = [2.6146035093439, 2.6146035093439]
        coeffs_imag = [-82.7871925498833,82.7871925498833]
        poles_real = [-4.649348606363277,-4.649348606363277]
        poles_imag = [7.142045840675953,-7.142045840675953]

        coeffs_real = [2.61, 2.61]
        coeffs_imag = [-82.78,82.78]
        poles_real = [-4.64,-4.64]
        poles_imag = [7.14,-7.14]

        #coeffs_real = [26, 26]
        #coeffs_imag = [-828,828]
        #poles_real = [-46,-46]
        #poles_imag = [71,-71]

        coeffs_real = [2, 2]
        coeffs_imag = [-1,1]
        poles_real = [-1,-1]
        poles_imag = [2,-2]
    else:

        coeffs_real =  \
            (100 * np.array([
            0.026146035093439,
            0.026146035093440,
            -3.781044669630893,
            -3.781044669630893,
            8.109797269074905,
            ])).tolist()
        coeffs_imag =  \
            (100 * np.array([
            -0.827871925498833,
            0.827871925498834,
            3.031174088800161,
            -3.031174088800161,
            0,
            ])).tolist()
        poles_real =  \
            np.array([
            -4.649348606363277,
            -4.649348606363277,
            -6.703912798307075,
            -6.703912798307075,
            -7.293477190659283,
            ]).tolist()
        poles_imag =  \
            np.array([
            7.142045840675953,
            -7.142045840675953,
            3.485322832366395,
            -3.485322832366395,
            0,
            ]).tolist()


    coeffs = tf.complex(real=tf.constant(coeffs_real, dtype=tf.float64),
                        imag=tf.constant(coeffs_imag, dtype=tf.float64))

    poles  = tf.complex(real=tf.constant(poles_real, dtype=tf.float64),
                        imag=tf.constant(poles_imag, dtype=tf.float64))

    if False:
        real = tf.constant([1], dtype=tf.float64)
        imag = tf.constant([0], dtype=tf.float64)

        ai = tf.complex(real, imag)
        bi = tf.complex(real, imag)
    else:
        ai = tf.reciprocal(coeffs)
        bi = tf.div(-poles, coeffs)

    print(sb.get_tensor_shape(ai))

    for i in range(sb.get_tensor_shape(ai)[0]):
        if pre_cond is None:
            # r = conjugate_gradient( Amatvec_old, b, x0, k=k)
            # r = conjugate_gradient( Amatvec, Bmatvec(b), x0, k=k) # this is also a correct use, missing mass mat.
            # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=Mmatvec) # but this one is better.
            # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=None)

            # print(Mmatvec(sd.batch_eye_like(b)), b)

            epsilon = 1e-8

            rhs = tf.linalg.solve(matrix=Mmatvec(sd.batch_eye_like(b))+sd.batch_eye_like(b)*epsilon, rhs=b)

            print(r2c(rhs))
            # rhs = b

            PP = r2c(P(sd.batch_eye_like(rhs)))
            II = r2c(Mmatvec(sd.batch_eye_like(rhs))+sd.batch_eye_like(rhs)*epsilon)

            r = c2r(tf.linalg.solve(
                matrix=PP*ai[i]+II*bi[i],
                rhs=r2c(Mmatvec(rhs)*tf.expand_dims(mesh.VM, -1))
            ))
        else:
            #r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_w, Bmatvec(b), x0, k=k)
            r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_s, Bmatvec(b), x0, k=k, Imatvec=Mmatvec)

        if i==0:
            tr = r
        else:
            tr = tr + r

    print('good so far')

    print('r shape:')

    tr = tf.multiply(tf.expand_dims(mesh.VM,-1), tr) # VM: [batch,n], b: [batch, n, c]

    print('r shape:')
    #print(r.get_shape())

    return tr


def heat_kernel_approx4(mesh, k, time, b=None, x0=None, pre_cond='spectral'):
    # this computes (M+time*L)^(-1)M*b

    b, x0 = heat_kernel_approx_pre(mesh, b=b, x0=x0)

    import surf_dg as sd
    if not hasattr(mesh, 'Fm'):
        [_, n, _] = sd.get_tensor_shape(mesh.V)
        mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n - 1)
    if not hasattr(mesh, 'DA'):
        mesh.DA = sd.per_face_doublearea(mesh.V, mesh.Fm)

    if not hasattr(mesh, 'Mass'):
        mesh.Mass = LumpMassDiagonalOperator(mesh)

    def Minv_matvec(U):
        return mesh.Mass.invmatvec(U, epsilon=1e-4, epsilonB=0)

    if False:
        # old implementation.
        # This is n.s.d.
        check_laplacian_dependencies(mesh)
        lap_op0 = SparseLaplacianOperator(mesh)
        mass_op0 = SparseFEMMass_v0(mesh)
        sign = -1
    else:
        lap_op0 = SparseFEMGalerkinLap(mesh,assembler=sd.lap_assembler_galerkin)
        # mass_op0 = SparseFEMGalerkinMass(mesh)
        mass_op0 = LumpMassDiagonalOperator(mesh)
        sign = 1

    # lap_op0 = LaplacianOperator(mesh)
    # mass_op0 = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    if pre_cond == 'spectral':

        if False:
            pre_op0_weak = LaplacianSpectralPreconditioner(mesh.A[:,:,0:40], mesh.b[:,0:40], skip_first=True)
            def Pmatvec_w(X):
                return pre_op0_weak.apply_to(X)
        if True:
            pre_op0_strong = LaplacianSpectralPreconditioner(mesh.A[:, :, 0:20], mesh.b[:, 0:20], skip_first=True)
            def Pmatvec_s(X):
                return pre_op0_strong.apply_to(X)

    else:
        assert pre_cond is None

    def Mmatvec(X):
        epsilon = 0 # 1e-8
        return mass_op0.apply_to(X) + epsilon * X


    if True:
        # Pade 3/3

        # (120+60x+12x^2+x^3)\(120-60x+12x^2-x^3)
        '''
        def Amatvec(X):
            epsilon = 1e-8  #
            # time = 0.001
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) + mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X

        def Bmatvec(X):
            epsilon = 1e-8
            return lap_op0.apply_to(lap_op0.apply_to(X) * (time * sign) - mass_op0.apply_to(X) * 6) * (time * sign) + mass_op0.apply_to(mass_op0.apply_to(X)) * 12 + epsilon * X
        '''

        def P(X):
            return lap_op0.apply_to(X) * (time * sign)

        def I(X):
            return mass_op0.apply_to(X)

        def Amatvec_0_0(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) + I(X) + epsilon * X

        def Bmatvec_0_0(X):
            epsilon = 1e-8
            return I(X) + epsilon * X

        # Pade 1/1
        # (1+0.5x)\(1-0.5x)
        def Amatvec_1_1(X):
            epsilon = 1e-8  #
            # time = 0.001
            return P(X) * 0.5 + I(X) + epsilon * X

        def Bmatvec_1_1(X):
            #epsilon = 1e-8
            return P(X) * (-0.5) + I(X) #+ epsilon * X # not necessary for B.

        Amatvec = Amatvec_1_1
        Bmatvec = Bmatvec_1_1


    #print('b shape:')
    #print(b.get_shape())

    #print('x0 shape:')
    #print(x0.get_shape())

    def dd(t):
        return tf.cast(t, tf.float64)
    def ss(t):
        return tf.cast(t, tf.float32)

    if False:

        ai = \
                (100 * np.array([
                -0.827871925498833,
                0.827871925498834,
                3.031174088800161,
                -3.031174088800161,
                0,
                ])).tolist()
    elif True:

        coeffs_real = \
            100 * np.array([
                0.026146035093439,
                0.026146035093440,
                -3.781044669630893,
                -3.781044669630893,
                8.109797269074905,
            ])
        coeffs_imag = \
            100 * np.array([
                -0.827871925498833,
                0.827871925498834,
                3.031174088800161,
                -3.031174088800161,
                0,
            ])
        poles_real = \
            np.array([
                -4.649348606363277,
                -4.649348606363277,
                -6.703912798307075,
                -6.703912798307075,
                -7.293477190659283,
            ])
        poles_imag = \
            np.array([
                7.142045840675953,
                -7.142045840675953,
                3.485322832366395,
                -3.485322832366395,
                0,
            ])

        ai = \
            np.array([
                1,
                1,
                1,
                1,
                0,
            ])

        bi = - 2 * (ai) * poles_real + (1-ai) * 1
        ci = ai * (poles_real**2 + poles_imag**2) - (1-ai) * poles_real
        di = ai * 2 * coeffs_real + (1-ai) * 0
        ei = ai * 2 * ( - poles_real * coeffs_real - coeffs_imag * poles_imag) + (1-ai) * coeffs_real

    else:
        ai = [0]
        bi = [1]
        ci = [1]
        di = [0]
        ei = [1]

    ai = tf.constant(ai, dtype=tf.float32)
    bi = tf.constant(bi, dtype=tf.float32)
    ci = tf.constant(ci, dtype=tf.float32)
    di = tf.constant(di, dtype=tf.float32)
    ei = tf.constant(ei, dtype=tf.float32)

    print(sb.get_tensor_shape(ai))

    for i in range(sb.get_tensor_shape(ai)[0]):
        if pre_cond is None:
            # r = conjugate_gradient( Amatvec_old, b, x0, k=k)
            # r = conjugate_gradient( Amatvec, Bmatvec(b), x0, k=k) # this is also a correct use, missing mass mat.
            # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=Mmatvec) # but this one is better.
            # r = conjugate_gradient(Amatvec, Bmatvec(b), x0, k=k, Imatvec=None)

            # print(Mmatvec(sd.batch_eye_like(b)), b)

            epsilon = 1e-3 # 1e-6

            rhs = P(Minv_matvec(b*di[i])) + ei[i] * b

            PP = P(sd.batch_eye_like(rhs))
            QQ = P(Minv_matvec(PP))
            II = Mmatvec(sd.batch_eye_like(rhs))

            matrix = QQ*ai[i]+PP*bi[i]+II*ci[i]+sd.batch_eye_like(rhs)*epsilon
            # return matrix

            r = tf.linalg.solve(
                matrix=dd(matrix),
                rhs=dd(rhs*tf.expand_dims(mesh.VM, -1)),
                #adjoint=True
            )
            r = ss(r)
        else:
            #r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_w, Bmatvec(b), x0, k=k)
            r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_s, Bmatvec(b), x0, k=k, Imatvec=Mmatvec)

        if i==0:
            tr = r
        else:
            tr = tr + r

    print('good so far')

    print('r shape:')

    tr = tf.multiply(tf.expand_dims(mesh.VM,-1), tr) # VM: [batch,n], b: [batch, n, c]

    print('r shape:')
    #print(r.get_shape())

    return tr


def heat_kernel_approx_quad(mesh, k, beta, gamma, b=None, x0=None):
    # 2(M+β*L) (M^2+2β*L*M+γ*L^2)^(-1)M*b

    check_laplacian_dependencies(mesh)
    if not hasattr(mesh, 'VM'):
        mesh.VM = vertex_mask(mesh.V)

    if b==None:
        [batch, n, _] = mesh.V.get_shape().as_list()
        # b = tf.zeros([batch, n, 1], tf.float32)
        b = tf.concat([tf.ones([batch, 1, 1], tf.float32), tf.zeros([batch, n-1, 1], tf.float32)], axis=1)
    elif b=='ones':
        [batch, n, _] = mesh.V.get_shape().as_list()
        b = tf.ones([batch, n, 1], tf.float32)

    b = tf.multiply(tf.expand_dims(mesh.VM,-1), b) # VM: [batch,n], b: [batch, n, c]

    if x0==None:
        x0 = init_guess_like(b)
    x0 = tf.multiply(tf.expand_dims(mesh.VM,-1), x0)

    # This is n.s.d.
    lap_op0 = LaplacianOperator(mesh)
    mass_op0 = DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

    pre_op0_weak = LaplacianSpectralPreconditioner(mesh.A[:,:,0:40], mesh.b[:,0:40], skip_first=True)
    pre_op0_strong = LaplacianSpectralPreconditioner(mesh.A[:, :, 50:90], mesh.b[:, 50:90], skip_first=True)

    #  (M^2+2β*L*M+γ*L^2)^(-1)
    def Amatvec(X):
        # note here L is p.s.d., amounts to -lap_op0
        #epsilon = 0 # 1e-6
        # time = 0.001
        MX = mass_op0.apply_to(X)
        return -lap_op0.apply_to( -lap_op0.apply_to(X) * gamma + MX * (2*beta) ) + mass_op0.apply_to(MX) #+ epsilon * X

    def Pmatvec_w(X):
        return pre_op0_weak.apply_to(X)

    def Pmatvec_s(X):
        return pre_op0_strong.apply_to(X)

    # 2(M+β*L)*M*b
    def Bmatvec(X):
        MX = mass_op0.apply_to(X)
        RX = mass_op0.apply_to(MX) * 2 - lap_op0.apply_to(MX) * (2 * beta)
        return RX

    r = preconditioned_conjugate_gradient(Amatvec, Pmatvec_w, Bmatvec(b), x0, k=k)

    r = tf.multiply(tf.expand_dims(mesh.VM,-1), r) # VM: [batch,n], b: [batch, n, c]

    return r


def model_mesh_mnist_101( U, mesh0, mesh1, mesh2, down01, down12):

    U01 = laplacian_filter(U, mesh0)
    # [batch, n0, c0]

    U02 = sn.flt(U01, 6, use_bias=True) + sn.flt(U, 6, use_bias=False)
    # [batch, n0, 6]

    U03 = tf.nn.relu(U02)

    U10 = pooling(U03, mesh0, mesh1, down01)
    # [batch, n1, 6]

    U11 = laplacian_filter(U10, mesh1)
    # [batch, n1, 6]

    U12 = sn.flt(U11, 16, use_bias=True) + sn.flt(U10, 16, use_bias=False)

    U13 = tf.nn.relu(U12)

    U20 = pooling(U13, mesh1, mesh2, down12)
    # [batch, n2, 6]

    fc0 = tf.layers.flatten(U20)
    # [batch, n2*16]

    #print(y.shape)

    mu = 0
    sigma = 0.1

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def vertex_mask(V):

    mask = tf.to_float(tf.logical_not(
        tf.logical_and(
            tf.logical_and(
                tf.less(tf.abs(V[:, :, 0]), 1e-9), tf.less(tf.abs(V[:, :, 1]), 1e-9)
            ), tf.less(tf.abs(V[:, :, 2]), 1e-9)
        )
    ))

    # print('mask shape=')
    # print(mask.get_shape())

    # [batch,n]
    return mask


def face_mask(F):

    # to apply the mask FM to a perface tensor TF of [batch,f,f_c]
    # tf.multiply(TF, tf.expand_dims(FM, -1))

    [batch, f, _] = F.get_shape().as_list()
    mask = tf.to_float( tf.logical_and( tf.greater_equal(F[:,:,0], 0), tf.less(F[:,:,0], f-1)) ) # tf.ones([batch,f])

    # the reason to require it strictly less than f-1 is due to unused entry could be initialized as f-1 as well.

    # [batch,f]
    return mask


def mesh_face_feature(mesh):

    def local_fmap(DE, W1, W2, W3, W4):

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

        F0 = DE  # F0 = tf.reshape(DE, [batch, f, 3])

        F1 = local_linear_trans(F0, 10, W1, use_bias=False)

        F2 = F1

        F3 = local_linear_trans(F2, 10, W2, use_bias=True)

        F4 = tf.nn.relu(F3) + F2

        F5 = local_linear_trans(F4, 10, W3, use_bias=True)

        F6 = tf.nn.relu(F5) + F4

        F7 = local_linear_trans(F6, 1, W4, use_bias=True)

        # [batch, f, 1]

        return tf.reshape(F7, [batch, f])

    if not hasattr(mesh, 'cot'):
        mesh.cot = sn.per_face_cot(mesh.V, mesh.F)

    '''
    def angle_from_edges(a, b, o):
        # the order of a,b does not matter.
        # o needs to be the edge opposite to the node.
        # assert a,b,o are of the same size.
        # the operation is applied element-wisely
        # the output have the same size as a,b,o.

        epsilon = 1e-10

        acos = tf.divide(tf.square(a) + tf.square(b) - tf.square(o), 2 * tf.multiply(a, b) + epsilon)

        return tf.acos(acos)

    if not hasattr(mesh, 'E'):
        mesh.E = sn.per_face_edges_length(mesh.V, mesh.F)

    if not hasattr(mesh, 'angle'):
        mesh.angle = tf.stack([
            angle_from_edges( mesh.E[:, :, 1],  mesh.E[:, :, 2],  mesh.E[:, :, 0]),
            angle_from_edges( mesh.E[:, :, 2],  mesh.E[:, :, 0],  mesh.E[:, :, 1]),
            angle_from_edges( mesh.E[:, :, 0],  mesh.E[:, :, 1],  mesh.E[:, :, 2])], axis=2)

    epsilon = 1e-10

    A = tf.reciprocal(tf.tan(mesh.angle) + epsilon)
    '''

    # now learn a 9 --> 1 mapping: fmap

    # in this way, the weights are shared in all fmaps.
    W1 = sb.weight_variable([3, 10])
    # W1 = tf.expand_dims( tf.constant([0.0,0.0,1.0]), -1)
    W2 = sb.weight_variable([10, 10])
    W3 = sb.weight_variable([10, 10])
    W4 = sb.weight_variable([10, 1])

    C = tf.stack([
        local_fmap(tf.gather(mesh.cot, [1, 2, 0], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [1, 2, 0], :]),
        local_fmap(tf.gather(mesh.cot, [2, 0, 1], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [2, 0, 1], :]),
        local_fmap(tf.gather(mesh.cot, [0, 1, 2], axis=2), W1, W2, W3, W4),  # fmap(DE[:, :, [0, 1, 2], :])
    ], axis=2)

    # DE[:, :, 1, :], DE[:, :, 2, :], DE[:, :, 0, :]
    # DE[:, :, 2, :], DE[:, :, 0, :], DE[:, :, 1, :]
    # DE[:, :, 0, :], DE[:, :, 1, :], DE[:, :, 2, :]

    # [batch, f, 3]
    return C


#def fem_assembler_lap(mesh, v0, v1, v2):
#    VC = mesh.V,
#    assembler =
#    return sd.lap_assembler(mesh.V[:,:,0], )


###########################

def augmentation(mesh, scale_tangent=0.0, scale_normal=0.0): # , scale_tangent=0.006, scale_normal=0.002, scale_lap=0.01):
    # import numpy
    # import numpy.random
    # print(numpy.random.rand(1))
    # import tensorflow
    # import tensorflow.random
    import tensorflow as tf

    [batch, n, _] = sd.get_tensor_shape(mesh.OV)
    scale_tangent = scale_tangent * 2
    scale_normal = scale_normal * 2

    if not hasattr(mesh, 'NV'):
        [batch, f, _] = sb.get_tensor_shape(mesh.F)
        WN = SparseFEMWedge(mesh, transpose=True).apply_to(U=tf.ones([batch, f, 1]))
        epsilon = 1e-6
        mesh.NV = WN / (tf.expand_dims(tf.norm(WN, axis=-1), axis=-1) + epsilon)

    T = scale_tangent * (tf.random_uniform(shape=[batch, n, 3]) - 0.5)
    T = T - tf.expand_dims(tf.reduce_sum(tf.multiply(mesh.NV, T), axis=-1), axis=-1) * T
    N = scale_normal * tf.multiply(mesh.NV, (tf.random_uniform(shape=[batch, n, 3]) - 0.5))

    # if scale_lap!= 0:
    #    lap_ori =

    RMV = mesh.V + T + N

    return RMV