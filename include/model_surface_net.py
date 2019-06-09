#

#class LearnLap():
#    def __init__(self):

# Afvs, Avfs = init(mesh, options, metadata)
def init(mesh, options, metadata):
    Afvs = [None for i in range(options.depth)]
    Avfs = [None for i in range(options.depth)]
    # L_matvec = [[] for i in range(options.depth)]
    for i in range(options.depth):
        slicer0 = options.learn_slicer(mesh)
        with tf.variable_scope('depth%d' % (i)) as sc:
            if i % 2 == 0:
                la0 = [
                    learn_assembler_wrapper(options.learn_assembler, out_dim=options.op_cdims[i],
                                            metadata=metadata,
                                            name='learn_assembler-%d-%d--0' % (i, jj)) for jj in range(1)]
                la1 = [
                    learn_assembler_wrapper(options.learn_assembler, out_dim=options.op_cdims[i],
                                            metadata=metadata,
                                            name='learn_assembler-%d-%d--1' % (i, jj)) for jj in range(1)]

                Afvs[i] = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=la0[jj],
                                               transpose=False) for jj in range(1)]
                Avfs[i] = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=la1[jj],
                                               transpose=True) for jj in range(1)]
                #    # L_matvec[i] = make(i)
                #    L_matvec[i] = make(i, mesh, options, metadata, flt, Minv_matvec, M_matvec)

    return Afvs, Avfs


def make_matvec(i, mesh, options, metadata, flt, Afv, Avf, Minv_matvec, M_matvec):
    import surf_basic as sb
    import math
    ceil = math.ceil

    def local_L_matvec(S):
        with tf.variable_scope('depth%d' % (i)) as sc:
            N1 = [[] for jj in range(1)]
            sf = 0
            for jj in range(1):
                # print('Layer input S:', S)
                # reduce the dimension.
                [_, _, sf] = sb.get_tensor_shape(S)
                S = flt(S, out_dim=ceil(sf / options.op_cdims[i]), name='reduce%d_f' % (i), reuse=True)
                # assert(options.num_ops[i]==1)
                # print('S reduced:', S)

                Bf = Afv[jj].apply_to(S)
                # print('Bf:',Bf)

                # mass matrix weighting
                Bf = tf.multiply(Bf, tf.expand_dims(mesh.DA, axis=-1))

                # reduce the dimension.
                [_, _, sv] = sb.get_tensor_shape(Bf)
                Bf = flt(Bf, out_dim=ceil(sv / options.op_cdims[i]), name='reduce%d_v' % (i), reuse=True)
                # print('Bf reduced:', Bf)

                Bv = Avf[jj].apply_to(Bf)
                # print('Bv:', Bv)


                # B = Bv + S # Residual connection.
                B = Bv
                # B = Minv_matvec(B)

                N1[jj] = Minv_matvec(B)

            N1 = tf.concat(N1, axis=-1)

            # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

            # if options.col_normalize:
            #    B = sb.batch_col_normalized(B, Imatvec=M_matvec)

            # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

            N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

            N2 = flt(N1, sf, name='AO%d_xx_0' % (i), reuse=True) # This is different to allow multiple forwarding!!
        return N2

    return local_L_matvec


def make_Di_matvec(i, mesh, options, metadata, flt, Afv, Avf, Minv_matvec, M_matvec):
    import surf_basic as sb
    import math
    ceil = math.ceil

    def local_Di_matvec(S):
        # effectively: 4f by 4n tensor.
        # input: [batch, 4n, sf]
        # [batch, n, 4*sf]
        # [batch, f, 4*sf*c]
        # [batch, 4f, sf*c]
        with tf.variable_scope('depth%d' % (i)) as sc:

            # print('Layer input S:', S)
            # reduce the dimension.
            [_, n4, sf] = sb.get_tensor_shape(S)
            assert (n4 % 4) == 0
            n = n4 // 4
            S = tf.reshape(S, shape=[-1,n,4*sf])

            S = flt(S, out_dim=4 * ceil(sf / options.op_cdims[i]), name='reduce%d_f' % (i), reuse=True)
            # assert(options.num_ops[i]==1)
            # print('S reduced:', S)

            N1 = [[] for jj in range(1)]
            for jj in range(1):
                Bf = Afv[jj].apply_to(S)
                # print('Bf:',Bf)

                Bf = flt(Bf, out_dim=4 * sf, name='reduce%d_f2' % (i), reuse=True)

                if False:
                    # mass matrix weighting
                    Bf = tf.multiply(Bf, tf.expand_dims(mesh.DA, axis=-1))

                N1[jj] = Bf

            N1 = tf.concat(N1, axis=-1)

            [_, f, ff] = sb.get_tensor_shape(N1)
            assert (ff%4)==0

            X = tf.reshape(N1, shape=[-1, f*4, ff//4])
        return X

    return local_Di_matvec


def make_DiA_matvec(i, mesh, options, metadata, flt, Afv, Avf, Minv_matvec, M_matvec):
    import surf_basic as sb
    import math
    ceil = math.ceil
    def local_DiA_matvec(S):
        # effectively: 4n by 4f tensor.
        # input: [batch, 4n, sf]

        with tf.variable_scope('depth%d' % (i)) as sc:

            [_, f4, sv] = sb.get_tensor_shape(S)
            assert (f4 % 4) == 0
            f = f4 // 4
            Bf = tf.reshape(S, shape=[-1, f, 4 * sv])

            if True:
                # mass matrix weighting
                Bf = tf.multiply(Bf, tf.expand_dims(mesh.DA, axis=-1))


            N1 = [[] for jj in range(1)]
            for jj in range(1):

                # reduce the dimension.
                Bf = flt(Bf, out_dim=4 * ceil(sv / options.op_cdims[i]), name='reduce%d_v' % (i), reuse=True)
                # print('Bf reduced:', Bf)

                Bv = Avf[jj].apply_to(Bf)
                # print('Bv:', Bv)

                Bv = flt(Bv, out_dim=4 * sv, name='reduce%d_v2' % (i), reuse=True)

                B = Bv
                # B = Minv_matvec(B)

                N1[jj] = Minv_matvec(B)

            N1 = tf.concat(N1, axis=-1)

            N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

            [_, n, nn] = sb.get_tensor_shape(N1)
            assert (n % 4) == 0

            X = tf.reshape(N1, shape=[-1, n * 4, nn // 4])

        return X

    return local_DiA_matvec


def model_surface_net(mesh_struct, metadata, options):

    import surf_util as su

    import surf_op as so
    import surf_dg as sd

    def assume_attr(str, default=su.NOT_GIVEN):
        su.assume_attr(options, str, default)

    # assume_attr('input_fun', input_fun_positions)
    assume_attr('type') # pointnet, lap, dir,
    assume_attr('depth', 15)

    assume_attr('op_cdims', [32 for i in range(options.depth)]) # important parameters to be given.
    assume_attr('learn_assembler', learn_assembler_1)  # learn_assembler_0
    assume_attr('learn_slicer', learn_slicer_local_0)  # learn_slicer_0

    import utils_fun as uf
    assume_attr('act_fn', uf.elu) # uf.relu

    options.model = options.type

    mesh0 = mesh_struct.mesh
    #mesh1 = mesh_struct.mesh_d
    #mesh2 = mesh_struct.mesh_d2
    #down01 = mesh_struct.ds01
    #down12 = mesh_struct.ds12

    print(options.name)


    def flt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=reuse)

    def nflt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=reuse)



    mesh = mesh0

    #if not hasattr(mesh, 'Fm'):
    [batch, n, _] = sd.get_tensor_shape(mesh.V)
    [batch, f, _] = sd.get_tensor_shape(mesh.F)

    mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n-1)
    # mesh.F = mesh.Fm

    import surf_dg as sd
    if not hasattr(mesh, 'DA'):
        mesh.DA = sd.per_face_doublearea(mesh.V, mesh.Fm)

    if not hasattr(mesh, 'Mass'):
        # mesh.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.Fm, mesh.J))
        mesh.Mass = sm.LumpMassDiagonalOperator(mesh)


    M_matvec = mesh.Mass.matvec
    def Minv_matvec(X):
        return mesh.Mass.invmatvec(X, epsilon=3e-4)


    # import surfacenet.models_tf as models
    import surfacenet.models_plus as models2

    U = mesh.A

    class BatchNorm1d():
        def __init__(self, size):
            #import torch.nn as nn
            #self.bn = nn.BatchNorm1d(size)

            BatchNorm1d.num_instances = BatchNorm1d.num_instances + 1

            return

        def forward(self, x):
            #self.bn.train()
            #x = self.bn(x)
            import tf_util
            return tf_util.batch_norm_for_fc(x, metadata.is_training,
                                          bn_decay=metadata.bn_decay, scope='Class-BatchNorm1d-%d'%BatchNorm1d.num_instances)

    BatchNorm1d.num_instances=0

    # model = models.AvgModel(batch_fun=BatchNorm1d)
    # U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U) #, same_op=True)

    if options.model in ["lap", "lap_lr", "nop"]:
        # model = models2.Model(batch_fun=BatchNorm1d) # original code, which blows up.
        model = models2.Model2(batch_fun=BatchNorm1d)
    elif options.model == "avg":
        model = models2.AvgModel(batch_fun=BatchNorm1d)
    elif options.model == "mlp":
        # model = models2.Model(batch_fun=BatchNorm1d) # confirmed, this does not work. It blows up.
        model = models2.MlpModel(batch_fun=BatchNorm1d) # original code.
    elif options.model in ["lape"]:
        model = models2.Model(batch_fun=BatchNorm1d) # original code, which blows up.
    else:
        assert options.model in ["dir", "dir_lr"]
        model = models2.DirModel2(batch_fun=BatchNorm1d)


    if False:

        Afv = sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=sd.grad_assembler, transpose=False)
        Avf = sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=sd.grad_assembler, transpose=True)

        import surf_basic as sb
        [_, n, _] = sb.get_tensor_shape(mesh.V)
        [_, f, _] = sb.get_tensor_shape(mesh.F)

        def Di_matvec(input):
            # effectively: 4f by 4n tensor.
            # input: [batch, 4n, sf]
            # [batch, n, 4*sf]
            # [batch, f, 4*sf*c]
            # [batch, 4f, sf*c]
            if True:
                print('Layer input S:', S)
                # reduce the dimension.
                [_, _, sf] = sb.get_tensor_shape(S)
                S = flt(S, out_dim=sf * options.Bf_rd[i], name='reduce%d_f' % (i), reuse=True)
                assert (options.num_ops[i] == 1)
                print('S reduced:', S)

            Bf = Afv[jj].apply_to(S)
            print('Bf:', Bf)

        def DiA_matvec(input):
            # mass matrix weighting
            Bf = tf.multiply(Bf, tf.expand_dims(mesh.DA, axis=-1))

            if True:
                # reduce the dimension.
                [_, _, sv] = sb.get_tensor_shape(Bf)
                Bf = flt(Bf, out_dim=sv * options.Bv_rd[i], name='reduce%d_v' % (i), reuse=True)
                print('Bf reduced:', Bf)

            Bv = Avf[jj].apply_to(Bf)
            print('Bv:', Bv)


    if options.model == "lap_lr":
        Afvs, Avfs = init(mesh, options, metadata)
        L_matvec = [make_matvec(ii, mesh, options, metadata, flt, Afvs[ii], Avfs[ii], Minv_matvec, M_matvec) if (ii%2)==0 else None for ii in range(options.depth)]
        U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U, same_op=False, is_training=metadata.is_training, act_fn=options.act_fn)
    elif options.model == "lap":
        A = sm.SparseFEMGalerkinLap(mesh)
        def L_matvec(x):
            # their lap is twice as large as mine.
            return 2 * Minv_matvec(A.apply_to(x))
        U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U, same_op=True, is_training=metadata.is_training, act_fn=options.act_fn)
    elif options.model == "nop":
        def L_matvec(x):
            return x
        U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U, same_op=True, is_training=metadata.is_training, act_fn=options.act_fn)
    elif options.model == "dir_lr":
        Afvs, Avfs = init(mesh, options, metadata)
        Di_matvec  = [make_Di_matvec(ii, mesh, options, metadata, flt, Afvs[ii], Avfs[ii], Minv_matvec, M_matvec) if (ii%2)==0 else None for ii in range(options.depth)]
        DiA_matvec = [make_DiA_matvec(ii, mesh, options, metadata, flt, Afvs[ii], Avfs[ii], Minv_matvec, M_matvec) if (ii%2)==0 else None for ii in range(options.depth)]
        U_out = model.forward(Di=Di_matvec, DiA=DiA_matvec, mask=mesh.Mask, inputs=U, num_faces=f, same_op=False, is_training=metadata.is_training, act_fn=options.act_fn)
    elif options.model == "dir":
        if True:
            def Di_matvec(x):
                import surf_basic as sb
                return sb.batch_sparse_matmul(mesh.Di, x)
            def DiA_matvec(x):
                import surf_basic as sb
                return sb.batch_sparse_matmul(mesh.DiA, x)
        else:
            Di = sm.SparseDi(mesh, transpose=False)
            DiA = sm.SparseDi(mesh, transpose=True)
            Di_matvec = Di.apply_to
            DiA_matvec = DiA.apply_to
        U_out = model.forward(Di=Di_matvec, DiA=DiA_matvec, mask=mesh.Mask, inputs=U, num_faces=f, same_op=True, is_training=metadata.is_training, act_fn=options.act_fn)
    elif options.model in ["lape", "avge", "mlpe"]:
        assert options.model == "lape"
        if True:
            A = sm.SparseFEMGalerkinLap(mesh)
            def L_matvec(x):
                # their lap is twice as large as mine.
                return 2*Minv_matvec(A.apply_to(x))
                # def L_matvec(x): # MLP
                #    return x
        else:
            def L_matvec(x):
                import surf_basic as sb
                return sb.batch_sparse_matmul(mesh.L, x)
        U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U, same_op=True)

    elif options.model in ["lap", "avg", "mlp"]:
        assert options.model == "mlp"
        if True:
            A = sm.SparseFEMGalerkinLap(mesh)
            def L_matvec(x):
                # their lap is twice as large as mine.
                return 2*Minv_matvec(A.apply_to(x))
                # def L_matvec(x): # MLP
                #    return x
        else:
            def L_matvec(x):
                import surf_basic as sb
                return sb.batch_sparse_matmul(mesh.L, x)
        U_out = model.forward(L=L_matvec, mask=mesh.Mask, inputs=U, same_op=True)
    else:
        assert False
        assert options.model == "dir"
        U_out = model.forward(Di=Di_matvec, DiA=DiA_matvec, mask=mesh.Mask, inputs=U, num_faces=f, same_op=True)

    U_out = U_out * mesh.Mask

    p_list = model.parameters()

    # print(p_list)
    # print(len(p_list))
    import numpy as np
    def get_size(p):
        # better solutions:
        # https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        # https://stackoverflow.com/questions/51509528/how-to-get-tensorflow-tensor-size-in-bytes
        return np.prod(p.get_shape().as_list())
    print('Model parameters: %d' % sum([get_size(p) for p in p_list]))


    # https://pytorch.org/docs/master/_modules/torch/optim/sgd.html#SGD
    # output = sn.ModelOutput(U_out, weights_l2_sum=tf.add_n([tf.nn.l2_loss(p) for p in p_list]))
    output = sn.ModelOutput(U_out, weights_l2_sum=tf.add_n([0.5*tf.reduce_sum(p*p) for p in p_list]))

    return output


from model_cls_lap import *


def model_surface_net_old( mesh_struct, metadata, options):

    import surf_util as su

    import surf_op as so
    import surf_dg as sd

    def assume_attr(str, default=su.NOT_GIVEN):
        su.assume_attr(options, str, default)

    assume_attr('out_type', 'cls')


    assume_attr('pooling')
    assume_attr('name')
    assume_attr('keep', '')
    # did not use this: assume_attr('per_face_feature_fun', sn.per_face_cot)
    assume_attr('input_fun', input_fun_positions)
    assume_attr('num_ops', [1, 1, 1])

    # These are for learnable operators
    assume_attr('op_cdims', [32, 32, 32]) # important parameters to be given.
    assume_attr('Bf_rd', [1, 1/32, 1/32])
    assume_attr('Bv_rd', [1/32, 1/32, 1/32])

    assume_attr('dims', [None, None, None])
    assume_attr('refresh', [True,True,True])
    assume_attr('mlp_dims', [512,256,128])
    assume_attr('k', 1)
    assume_attr('keep_old_feature', True)
    #assume_attr('assembler', assembler_20x_1)
    #assume_attr('assembler', sn.lap_assembler_v0)
    #assume_attr('assembler_feature', assembler_feature_default)
    #assume_attr('assembler_mesh', None) # this will overwrite the assembler.
    assume_attr('learn_slicer', None)  # learn_slicer_0
    assume_attr('learn_assembler', None) # learn_assembler_0
    assume_attr('col_normalize', True)
    assume_attr('setup_type', '2')

    assume_attr('down', [False for ii in range(len(options.refresh))])

    mesh0 = mesh_struct.mesh
    mesh1 = mesh_struct.mesh_d
    mesh2 = mesh_struct.mesh_d2
    #down01 = mesh_struct.ds01
    #down12 = mesh_struct.ds12

    #if options.assembler_mesh is not None:
    #    def assembler(v0, v1, v2):
    #        return options.assembler_mesh(mesh0, v0, v1, v2, metadata)
    #    options.assembler = assembler


    print(options.name)

    U = options.input_fun(mesh0)

    #import surf_dg as sd
    #lap_op0 = sm.SparseLaplacianOperator(mesh0, per_face_feature_fun=sd.per_face_cot)

    #mass_op0 = sm.DiagonalOperator(sn.mass_barycentric(mesh0.V, mesh0.F, mesh0.J))
    # - t * lap_op0.ops.to_dense()

    assert options.refresh[0]

    current_mesh_level = -1

    def flt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=reuse)

    def nflt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=reuse)

    if True:

        # Initialize here:

        VC = mesh0.V
        S = U

        for i in range(len(options.dims)):

            with tf.variable_scope('depth%d'%(i)) as sc:

                if i==0 or (i>0 and options.down[i-1] is True):
                    current_mesh_level = current_mesh_level + 1
                    assert options.refresh[i] is True
                    if current_mesh_level==0:
                        mesh = mesh0
                        check_K(mesh0)
                    elif current_mesh_level==1:
                        mesh = mesh1
                        check_K(mesh1)
                    elif current_mesh_level==2:
                        mesh = mesh2
                        check_K(mesh2)
                    else:
                        assert False

                    # clear any operator if created.
                    A = []
                    Afv = []
                    Avf = []

                    # TODO: downsample S and set up VC

                    if not hasattr(mesh, 'Mass'):
                        mesh.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.F, mesh.J))

                    import surf_dg as sd
                    if not hasattr(mesh, 'DA'):
                        mesh.DA = sd.per_face_doublearea(mesh.V, mesh.F)

                    M_matvec = mesh.Mass.matvec
                    Minv_matvec = mesh.Mass.invmatvec


                # in: S, VC
                # out: S, VC

                if options.setup_type == 'pointnet':

                    S = flt(S, options.dims[i], name='AO%d_1' % i)

                elif options.setup_type=='0':
                    # Fixed Laplacian Operator

                    if options.refresh[i] or i==0:
                        #with tf.variable_scope('assembler_feature-%d'%(i)):
                        #    # assembler_feature_default may need a new scope
                        #    tVC = options.assembler_feature(mesh, VC, metadata)
                        #with tf.variable_scope('assembler-%d' % (i)):
                        #    A = sm.SparseFEMGeneral(mesh, VC=tVC, assembler=options.assembler)
                        assert options.num_ops[i]==1

                        #A = sm.SparseFEMGeneral_v0(mesh, VC=assembler_feature_default(mesh, VC, metadata),
                        #                        assembler=options.assembler)

                        A = sm.SparseFEMGalerkinLap(mesh)

                    # the graph conv

                    def graph_conv1x1(num_inputs, num_outputs, batch_norm=None):
                        # todo
                        return


                    if i==0:
                        assert options.dims[i]==128
                        S = flt(S, 128) #todo: no batch_norm








                    N1 = Minv_matvec(A.apply_to(S))
                    # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

                    N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

                    X1 = flt(N1, options.dims[i], name='AO%d_0'%i)
                    X2 = nflt(S, options.dims[i], name='AO%d_1'%i)
                    B = X1 + X2
                    #B = apply_operator(S, A, M_matvec, Minv_matvec, options.dims[i], metadata.is_training,
                    #                   metadata.bn_decay, keep=options.keep)  # c1=10

                    # P =

                    S = flt(S, out_dim=options.dims[i], name='S%d_1' % (i)) \
                        + flt(B, out_dim=options.dims[i], name='S%d_2' % (i))  # tf.concat([, VC], axis=-1)

                    VC = flt(S, out_dim=options.dims[i], name='VC%d_1' % (i)) \
                         + flt(B, out_dim=options.dims[i], name='VC%d_2' % (i))

                elif options.setup_type=='1':

                    # Fixed Laplacian Operator, inside apply operator.

                    if options.refresh[i] or i==0:
                        #with tf.variable_scope('assembler_feature-%d'%(i)):
                        #    # assembler_feature_default may need a new scope
                        #    tVC = options.assembler_feature(mesh, VC, metadata)
                        #with tf.variable_scope('assembler-%d' % (i)):
                        #    A = sm.SparseFEMGeneral(mesh, VC=tVC, assembler=options.assembler)

                        assert options.num_ops[i] == 1

                        A = sm.SparseFEMGeneral_v0(mesh, VC=assembler_feature_default(mesh, VC, metadata),
                                                assembler=options.assembler)


                    B = apply_operator(S, A, M_matvec, Minv_matvec, options.dims[i], metadata.is_training,
                                       metadata.bn_decay, keep=options.keep)  # c1=10

                    # P =

                    S = flt(S, out_dim=options.dims[i], name='S%d_1' % (i)) + flt(B, out_dim=options.dims[i],
                                                                                  name='S%d_2' % (
                                                                                  i))  # tf.concat([, VC], axis=-1)

                    VC = flt(S, out_dim=options.dims[i], name='VC%d_1' % (i)) + flt(B, out_dim=options.dims[i],
                                                                                    name='VC%d_2' % (i))

                else:

                    assert options.setup_type[0]=='2' \
                           or options.setup_type[0]=='3' \
                           or options.setup_type[0]=='4'

                    if options.refresh[i] or i==0: # set operators.

                        assert options.num_ops[i]==1

                        if options.setup_type[0]=='2':  # Fixed Grad Operator.
                            #assembler_feature_m()

                            if i == 0:
                                slicer0 = sd.grad_slicer(mesh.V, mesh.F)

                            #assert options.num_ops[i]==1 # no sense to use more than one fixed operators.
                            if options.num_ops[i]!=1:
                                print('Warning: no sense to use more than one fixed operators.')

                            Afv = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=sd.grad_assembler, transpose=False) for jj in range(options.num_ops[i])]
                            Avf = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=sd.grad_assembler, transpose=True) for jj in range(options.num_ops[i])]
                        else:
                            assert options.setup_type[0]=='3' # Learn-able Rect Operator.

                            if i == 0:
                                slicer0 = options.learn_slicer(mesh)

                            # options.learn_assembler:
                            if not options.learn_assembler == learn_assembler_1:
                                la0 = [learn_assembler_wrapper(options.learn_assembler, out_dim=options.op_cdims[i], metadata=metadata, name='learn_assembler-%d-%d--0' % (i,jj)) for jj in range(options.num_ops[i])]
                                la1 = [learn_assembler_wrapper(options.learn_assembler, out_dim=options.op_cdims[i], metadata=metadata, name='learn_assembler-%d-%d--1' % (i,jj)) for jj in range(options.num_ops[i])]
                                # Critical to use different name for each jj, becauses weights are not shared between operators.
                            else:
                                la0 = [] #[[] in range(options.num_ops[i])]:
                                la1 = [] #[[] in range(options.num_ops[i])]
                                assert options.learn_assembler == learn_assembler_1
                                for jj in range(options.num_ops[i]):
                                    def learn_assembler_inline_0(vafc0, vafc1, vafc2, ec0, ec1, ec2): #, metadata, nname):
                                        # it will be great not to use vafc1, vafc2

                                        with tf.variable_scope('learn_assembler-%d-%d--0'%(i,jj)) as sc:

                                            reuse = True # has to reuse weights.

                                            dim1 = 32

                                            out = nflt(ec0, out_dim=dim1, name='01', reuse=reuse) \
                                                  + nflt(ec1, out_dim=dim1, name='02', reuse=reuse) + nflt(ec2, out_dim=dim1, name='03', reuse=reuse)
                                            # flt(vafc0, out_dim=dim1, name='00', reuse=reuse) +

                                            dim2 = 32

                                            out = tf.nn.relu(flt(out, out_dim=dim2, name='2', reuse=reuse))

                                            dim3 = 32

                                            out = tf.nn.relu(flt(out, out_dim=dim3, name='3', reuse=reuse))

                                            out = flt(out, out_dim=options.op_cdims[i],
                                                      name='4', reuse=reuse)  # no relu in the last to allow negative weights.

                                        # [batch, f, f_c=3]
                                        return out

                                    def learn_assembler_inline_1(vafc0, vafc1, vafc2, ec0, ec1, ec2): #, metadata, nname):
                                        # it will be great not to use vafc1, vafc2

                                        with tf.variable_scope('learn_assembler-%d-%d--1'%(i,jj)) as sc:

                                            reuse = True # has to reuse weights.

                                            dim1 = 32

                                            out = nflt(ec0, out_dim=dim1, name='01', reuse=reuse) \
                                                  + nflt(ec1, out_dim=dim1, name='02', reuse=reuse) + nflt(ec2, out_dim=dim1, name='03', reuse=reuse)
                                            # flt(vafc0, out_dim=dim1, name='00', reuse=reuse) +

                                            dim2 = 32

                                            out = tf.nn.relu(flt(out, out_dim=dim2, name='2', reuse=reuse))

                                            dim3 = 32

                                            out = tf.nn.relu(flt(out, out_dim=dim3, name='3', reuse=reuse))

                                            out = flt(out, out_dim=options.op_cdims[i],
                                                      name='4', reuse=reuse)  # no relu in the last to allow negative weights.

                                        # [batch, f, f_c=3]
                                        return out
                                    la0.append(learn_assembler_inline_0) # la0[jj] = learn_assembler_0
                                    la1.append(learn_assembler_inline_1) # la1[jj] = learn_assembler_1

                            Afv = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=la0[jj],
                                                      transpose=False) for jj in range(options.num_ops[i])]
                            Avf = [sm.SparseFEMGeneral(mesh, slicer=slicer0, assembler=la1[jj],
                                                      transpose=True) for jj in range(options.num_ops[i])]

                    # print('240: S',S.shape)

                    N1 = [[] for jj in range(options.num_ops[i])]
                    for jj in range(options.num_ops[i]):

                        if True:
                            print('Layer input S:', S)
                            # reduce the dimension.
                            [_, _, sf] = sb.get_tensor_shape(S)
                            S = flt(S, out_dim=sf*options.Bf_rd[i], name='reduce%d_f' % (i), reuse=True)
                            assert(options.num_ops[i]==1)
                            print('S reduced:', S)

                        Bf = Afv[jj].apply_to(S)
                        print('Bf:',Bf)

                        # mass matrix weighting
                        Bf = tf.multiply( Bf, tf.expand_dims(mesh.DA, axis=-1))

                        if True:
                            # reduce the dimension.
                            [_, _, sv] = sb.get_tensor_shape(Bf)
                            Bf = flt(Bf, out_dim=sv*options.Bv_rd[i], name='reduce%d_v' % (i), reuse=True)
                            print('Bf reduced:', Bf)

                        Bv = Avf[jj].apply_to(Bf)
                        print('Bv:', Bv)


                        #B = Bv + S # Residual connection.
                        B = Bv
                        # B = Minv_matvec(B)

                        N1[jj] = Minv_matvec(B)

                    N1 = tf.concat(N1, axis=-1)

                    # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

                    #if options.col_normalize:
                    #    B = sb.batch_col_normalized(B, Imatvec=M_matvec)

                    # allowing multiple ops: Z1 = tf.concat([A.apply_to(U[:,:,(4*i):(min(4*i+4,c))]) for i in range((c-1)//4+1)], axis=-1)

                    N1 = sb.batch_col_normalized(N1, Imatvec=M_matvec)

                    X1 = N1 if options.dims[i] is None \
                        else flt(N1, options.dims[i], name='AO%d_0' % i)
                    X2 = nflt(S, sb.get_tensor_shape(X1)[2], name='AO%d_1' % i) if options.dims[i] is None \
                        else nflt(S, options.dims[i], name='AO%d_1' % i)
                    B = X1 + X2

                    print('B:', B)

                    # B = apply_operator(S, A, M_matvec, Minv_matvec, options.dims[i], metadata.is_training, metadata.bn_decay, keep=options.keep)  # c1=10
                    #

                    #if options.block_type == 2:
                    #    B = flt(B, out_dim=options.dims[i], name='B%d_1' % (i)) + nflt(B, out_dim=options.dims[i],
                    #                                                                   name='B%d_2' % (i))

                    # P =

                    if False: # I think this should be redudant.
                        S = flt(S, out_dim=options.dims[i], name='S%d_1' % (i)) + nflt(B, out_dim=options.dims[i],name='S%d_2' % (i))  # tf.concat([, VC], axis=-1)
                    else:
                        S = B

                    print('S:', S)

                    if False:
                        VC = flt(S, out_dim=options.dims[i], name='VC%d_1' % (i)) \
                             + nflt(B, out_dim=options.dims[i], name='VC%d_2' % (i))

        U_out = S

    # fc0 = tf.layers.flatten(U_out)


    if options.out_type == 'cls':

        assume_attr('num_class')

        import surf_pointnet as sp
        logits = sp.multiple_linear_perpecton(max_pooling(U_out), mlp_dims=options.mlp_dims, is_training=metadata.is_training, bn_decay=metadata.bn_decay, num_class=options.num_class)
        # logits = sp.multiple_linear_perpecton(max_pooling(U_out), mlp_dims=[1, 1, 1], is_training=True, bn_decay=None, num_class=NUM_CLASS)

        output = sn.ModelOutput(logits)

    elif options.out_type == 'cls-dense':

        assume_attr('num_class')

        from tensorflow.contrib.layers import flatten
        import surf_pointnet as sp
        logits = sp.multiple_linear_perpecton(flatten(U_out), mlp_dims=options.mlp_dims, is_training=metadata.is_training, bn_decay=metadata.bn_decay, num_class=options.num_class)

        output = sn.ModelOutput(logits)

    else:
        assert options.out_type == 'attr'

        output = sn.ModelOutput(U_out)

    return output

