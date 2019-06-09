from model_cls_lap import *

# this will overwrite the model


def model_cls_lap_24X( mesh_struct, metadata, options):

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

    assume_attr('layer_gpu', [None for ii in range(len(options.dims))])
    assume_attr('do_inner_prod', [False for ii in range(len(options.dims))])
    if options.do_inner_prod is True or options.do_inner_prod is False:
        # this makes sense, since [True] does not equal to either.
        v = options.do_inner_prod is True
        options.do_inner_prod = [v for ii in range(len(options.dims))]


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
    #assume_attr('vertex_pertubation', None)
    assume_attr('augmentor', None) # sm.augmentation

    assume_attr('down', [False for ii in range(len(options.refresh))])

    mesh0 = mesh_struct.mesh
    #mesh1 = mesh_struct.mesh_d
    #mesh2 = mesh_struct.mesh_d2
    #down01 = mesh_struct.ds01
    #down12 = mesh_struct.ds12

    #if options.assembler_mesh is not None:
    #    def assembler(v0, v1, v2):
    #        return options.assembler_mesh(mesh0, v0, v1, v2, metadata)
    #    options.assembler = assembler


    print(options.name)


    #import surf_dg as sd
    #lap_op0 = sm.SparseLaplacianOperator(mesh0, per_face_feature_fun=sd.per_face_cot)

    #mass_op0 = sm.DiagonalOperator(sn.mass_barycentric(mesh0.V, mesh0.F, mesh0.J))
    # - t * lap_op0.ops.to_dense()

    assert options.refresh[0]

    def flt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=True, name=name, reuse=reuse)

    def nflt(XXX, out_dim, name=None, reuse=False):
        return sn.flt(XXX, out_dim,
                      metadata.is_training, metadata.bn_decay, use_bias=False, name=name, reuse=reuse)

    current_mesh_level = -1


    #if options.vertex_pertubation is not None:
    if options.augmentor is not None:
        mesh0.V_ori = mesh0.OV
        [_, n, _] = sd.get_tensor_shape(mesh0.OV)
        mesh0.Fm = sd.change_padding_value(mesh0.F, padvalue=-1, newvalue=n - 1)
        mesh0.V = mesh0.OV # have to do it so augmentor pass to compute normal.
        w = tf.cond(metadata.is_training,
                               lambda: 1.0,
                               lambda: 0.0)
        mesh0.V = w * options.augmentor(mesh0) + (1-w) * mesh0.OV
    else:
        if hasattr(mesh0, 'OV'):
            # compatible with old version.
            mesh0.V = mesh0.OV

    # this should start after augmentation.
    U = options.input_fun(mesh0, metadata)

    if True:

        # Initialize here:

        VC = mesh0.V
        S = U

        for i in range(len(options.dims)):
            #'''
            #device_handle = None
            if options.layer_gpu[i] is not None:
                device = tf.device('/gpu:' + str(options.layer_gpu[i]))
                device_handle = device.__enter__()
            #'''

            with tf.variable_scope('depth%d'%(i)) as sc:

                if i==0 or (i>0 and options.down[i-1] is True):
                    current_mesh_level = current_mesh_level + 1
                    assert options.refresh[i] is True
                    if current_mesh_level==0:
                        mesh = mesh0
                        #check_K(mesh0)
                    elif current_mesh_level==1:
                        mesh1 = mesh_struct.mesh_d
                        down01 = mesh_struct.ds01
                        mesh = mesh1
                        #check_K(mesh1)
                    elif current_mesh_level==2:
                        mesh2 = mesh_struct.mesh_d2
                        down12 = mesh_struct.ds12
                        mesh = mesh2
                        #check_K(mesh2)
                    else:
                        assert False
                    # clear any operator if created.
                    A = []
                    Afv = []
                    Avf = []

                    # TODO: downsample S and set up VC

                    #if not hasattr(mesh, 'Fm'):
                    [_, n, _] = sd.get_tensor_shape(mesh.V)
                    mesh.Fm = sd.change_padding_value(mesh.F, padvalue=-1, newvalue=n-1)
                    # mesh.F = mesh.Fm

                    if not hasattr(mesh, 'Mass'):
                        # mesh.Mass = sm.DiagonalOperator(sn.mass_barycentric(mesh.V, mesh.Fm, mesh.J))
                        mesh.Mass = sm.LumpMassDiagonalOperator(mesh)

                    import surf_dg as sd
                    if not hasattr(mesh, 'DA'):
                        mesh.DA = sd.per_face_doublearea(mesh.V, mesh.Fm)

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

                        A = sm.SparseFEMGeneral_v0(mesh, VC=assembler_feature_default(mesh, VC, metadata),
                                                assembler=options.assembler)


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
                                slicer0 = sd.grad_slicer(mesh.V, mesh.Fm)

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

                    if options.do_inner_prod[i]:
                        [_, n, _] = sb.get_tensor_shape(B)
                        B = tf.concat([B,sb.batch_spread(sb.batch_inner_keep(X1, X2, Imatvec=M_matvec), multiples=n)], axis=-1)

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

            '''
            if options.layer_gpu[i] is not None:
                device_handle.__exit__(None, None, None)
            '''

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

