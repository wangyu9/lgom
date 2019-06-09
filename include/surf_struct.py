import surf_util as su
import tensorflow as tf
import numpy as np


MODELNET40_CLASS = \
['airplane',
 'bathtub',
 'bed',
 'bench',
 'bookshelf',
 'bottle',
 'bowl',
 'car',
 'chair',
 'cone',
 'cup',
 'curtain',
 'desk',
 'door',
 'dresser',
 'flower_pot',
 'glass_box',
 'guitar',
 'keyboard',
 'lamp',
 'laptop',
 'mantel',
 'monitor',
 'night_stand',
 'person',
 'piano',
 'plant',
 'radio',
 'range_hood',
 'sink',
 'sofa',
 'stairs',
 'stool',
 'table',
 'tent',
 'toilet',
 'tv_stand',
 'vase',
 'wardrobe',
 'xbox']


def data_tensors(allV, allF, allA, allb, allJ, allK, n=800, f=1600, a=16, names=None):
    Vs = []
    Fs = []
    As = []
    bs = []
    Js = []
    Ks = []
    for i in range(0, len(allV)):
    # for i in range(0, allV.shape[0]):
        V, F = allV[i], allF[i]
        A = allA[i]
        b = allb[i]
        Vs.append(V)
        Fs.append(F)
        As.append(A)
        bs.append(b)
        Js.append(allJ[i])
        Ks.append(allK[i])

    # Js, Ks = su.meshes_adjacenies(Vs, Fs, a=a, names=names)

    TV, TF, TJ, TK = su.meshes_to_tensor(Vs, Fs, Js, Ks, n, f, a, minus_unused_indices=False)
    TA = su.meshes_attributes_to_tensor(As, n)

    Tb = su.meshes_info_to_tensor(bs)

    return TV, TF, TJ, TK, TA, Tb


# TODO: use the same n,f,a for data_tensor
def data_tensors_old(allV, allF, allA, allb, n=800, f=1600, a=16, names=None):
    Vs = []
    Fs = []
    As = []
    bs = []
    # namesX = []
    for i in range(0, len(allV)):
    # for i in range(0, allV.shape[0]):
        V, F = allV[i], allF[i]
        A = allA[i]
        b = allb[i]
        Vs.append(V)
        Fs.append(F)
        As.append(A)
        bs.append(b)
        #if names==None:
        #    #namesX.append('')
        #    print('No Mesh Name')
        #else:
        #    #namesX.append(allName[i])
        #    print('Mesh[%5d] name: %s'%(i,names[i]))

    # print("Converting mesh data to tensors...")
    # print(max([Vs[i].shape[0] for i in range(len(Vs))]))
    # print(max([Fs[i].shape[0] for i in range(len(Fs))]))

    Js, Ks = su.meshes_adjacenies(Vs, Fs, a=a, names=names)

    TV, TF, TJ, TK = su.meshes_to_tensor(Vs, Fs, Js, Ks, n, f, a, minus_unused_indices=False)
    TA = su.meshes_attributes_to_tensor(As, n)

    Tb = su.meshes_info_to_tensor(bs)

    return TV, TF, TJ, TK, TA, Tb


def data_attr_tensors(allYVA, allYFA, n=800, f=1600):
    YVAs = []
    YFAs = []
    for i in range(0, len(allYVA)):
        YVAs.append(allYVA[i])
        YFAs.append(allYFA[i])

    TYVA = su.meshes_attributes_to_tensor(YVAs, n)
    TYFA = su.meshes_attributes_to_tensor(YFAs, f)

    return TYVA, TYFA


def tf_sparse_placeholder(batch, n, a, name):
    b = tf.placeholder(tf.int64, shape=[None], name=name+'/b')
    r = tf.placeholder(tf.int64, shape=[None], name=name+'/r')
    c = tf.placeholder(tf.int64, shape=[None], name=name+'/c')
    v = tf.placeholder(tf.int64, shape=[None], name=name+'/v')

    K = tf.SparseTensor(indices=tf.stack([b,r,c],axis=-1), values=v, dense_shape=[batch,n,a]) # cannot have name , name=name)

    return K, {'b':b, 'r':r, 'c':c, 'v':v}


def tf_sparse_feed_list(h, out, name):

    print('tf_sparse_feed_list',
          out[name + '/b'].shape,
          out[name + '/r'].shape,
          out[name + '/c'].shape,
          out[name + '/v'].shape)

    feed_list = {
        h['b']: out[name+'/b'],
        h['r']: out[name+'/r'],
        h['c']: out[name+'/c'],
        h['v']: out[name+'/v']
    }
    return feed_list


def tf_trans(K, dict, fun):

    shape = K.get_shape().as_list()
    return tf.SparseTensor(indices=[dict['b'],dict['r'],dict['c']], values=fun(dict['v']), dense_shape=shape)


###################################################


def default_create_mesh_tf(batch, provider):
    # return create_mesh_core(batch, self.provider.n, self.provider.f, self.provider.a, self.provider.ta, self.provider.tb, prefix=self.name+'/', use_sparse=self.provider.use_sparse)
    return _create_mesh_core(batch, provider.n, provider.f, provider.a, provider.ta,
                            provider.tb, prefix='', use_sparse=provider.use_sparse)


def _create_mesh_core(batch, n, f, a, ta, tb, prefix, use_sparse):

    mesh = su.C()

    mesh.V = tf.placeholder(tf.float32, shape=[batch, n, 3], name='V')
    mesh.F = tf.placeholder(tf.int32, shape=[batch,f,3], name='F')

    if use_sparse:
        mesh.J = tf.sparse_placeholder(tf.int32, shape=[batch,n,a], name='J')
        mesh.K = tf.sparse_placeholder(tf.int32, shape=[batch,n,a], name='K')

        mesh.sJ, mesh.Jh = tf_sparse_placeholder(batch, n, a, name='J')
        mesh.sK, mesh.Kh = tf_sparse_placeholder(batch, n, a, name='K')

        import surf_dg as sd
        # mesh.K_succ = sd.succ(mesh.K)
        # mesh.K_prev = sd.prev(mesh.K)

        mesh.K_succ = tf_trans(mesh.K, mesh.Kh, sd.succ)
        mesh.K_prev = tf_trans(mesh.K, mesh.Kh, sd.prev)

        #if True:
            # TODO: Debugging purpose, remove later.
            # This is not exactly equivalent, since this pad 0 to the dense tensor instead of -1.
            #mesh.J = tf.sparse_tensor_to_dense(mesh.sJ)
            #mesh.K = tf.sparse_tensor_to_dense(mesh.sK)
            #mesh.K_succ = tf.sparse_tensor_to_dense(mesh.K_succ)
            #mesh.K_prev = tf.sparse_tensor_to_dense(mesh.K_prev)



    else:
        mesh.J = tf.placeholder(tf.int32, shape=[batch,n,a], name='J')
        mesh.K = tf.placeholder(tf.int32, shape=[batch,n,a], name='K')

    mesh.A = tf.placeholder(tf.float32, shape=[batch,n,ta], name='A')
    mesh.b = tf.placeholder(tf.float32, shape=[batch,tb], name='b')

    return mesh


def default_make_mesh_feed_list(tf_data, provider, out):
    if provider.use_sparse:
        feed_list = {
            tf_data.V: out['V'],
            tf_data.F: out['F'],
            #tf_data.J: out['J'],
            #tf_data.K: out['K'],
            tf_data.A: out['A'],
            tf_data.b: out['b']
        }
        feed_list.update(tf_sparse_feed_list(tf_data.Jh, out, 'J'))
        feed_list.update(tf_sparse_feed_list(tf_data.Kh, out, 'K'))
    else:
        feed_list = {
            tf_data.V: out['V'],
            tf_data.F: out['F'],
            tf_data.J: out['J'],
            tf_data.K: out['K'],
            tf_data.A: out['A'],
            tf_data.b: out['b']
        }
    return feed_list


class default_MeshBatch:

    def __init__(self, provider, out):
        if provider.use_sparse:
            # TODO
            self.TV = out['V']
            self.TF = out['F']
            #self.TJ = out['J']
            #self.TK = out['K']
            self.TA = out['A']
            self.Tb = out['b']
        else:

            # TV, TF, TJ, TK, TA, Tb
            self.TV = out['V']
            self.TF = out['F']
            self.TJ = out['J']
            self.TK = out['K']
            self.TA = out['A']
            self.Tb = out['b']

    #def make_feed_list(self, mesh):
    #    return make_mesh_feed_list(mesh.tf_data, self.TV, self.TF, self.TJ, self.TK, self.TA, self.Tb)


class default_MeshProvider:
    # n, f, a, ta, tb, use_sparse, get, getV
    def print(self, s):
        print('MeshProvider(): %s'%(s))

    def __init__(self, h5, phase, level, max_adj):
        self.h5 = h5
        self.phase = phase
        self.level = level
        self.use_sparse = False

        prefix = level_to_mesh_prefix(level) + phase

        def TR(variable):
            return TensorRegister(self.h5, prefix, variable)

        self.rV = TR('TV')
        self.rF = TR('TF')
        self.rA = TR('TA')
        self.rb = TR('Tb')
        # # note there is no first dim for batch!
        self.n = self.rV.shape[0]
        self.f = self.rF.shape[0]

        self.ta = self.rA.shape[1]
        self.tb = self.rb.shape[0]

        if self.use_sparse:
            # TODO
            self.a = 30 # An large enough number for now.

        else:

            self.rJ = TR('TJ')
            self.rK = TR('TK')
            self.a = self.rJ.shape[1]
            if not (max_adj is None):
                self.a = min(max_adj, self.a)

        self.print('%s n=%d, f=%d, a=%d, ta=%d, tb=%d'%(prefix,self.n,self.f,self.a,self.ta,self.tb))

        return

    def get(self, r, augmentor=None):

        if augmentor==None:
            augmentor = Augmentor()

        if self.use_sparse:
            TF =  self.rF.get(r)
            J_list, K_list = faces_to_adjacencies(TF, shape=[self.n, self.a])
            dict = {'V':augmentor.apply(modify_padding(self.rV.get(r))),
                    'F': TF, 'A':self.rA.get(r), 'b':self.rb.get(r)}
            dict.update(J_list)
            dict.update(K_list)
            return dict
        else:
            return {'V':augmentor.apply(modify_padding(self.rV.get(r))),
                    'F': self.rF.get(r), 'J': self.rJ.get(r)[:,:,0:self.a], 'K': self.rK.get(r)[:,:,0:self.a], 'A': self.rA.get(r), 'b':self.rb.get(r)}

    def getV(self, r):

        return modify_padding(self.rV.get(r))


class default_DownProvider0:

    def print(self, s):
        print('DownProvider(): %s'%(s))

    def __init__(self, h5, phase, level, mesh_provider=None, mesh_provider_d=None):
        self.h5 = h5
        self.phase = phase
        self.level = level

        prefix = level_to_down_prefix(level) + phase

        def TR(variable):
            return TensorRegister(self.h5, prefix, variable)


        self.rDSI = TR('TDSI')
        self.rDSW = TR('TDSW')

        # note there is no first dim for batch!
        self.n = self.rDSW.shape[0]
        self.k = self.rDSW.shape[1]
        assert(self.k==self.rDSI.shape[1])

        self.print('%s k=%d'%(prefix,self.k))

        '''
        self.mesh_provider = mesh_provider
        self.mesh_provider_d = mesh_provider_d
        '''

        return

    def get(self, r):

        '''
        h5f = self.h5 + '.part0.h5'
        phase = self.phase
        level = self.level

        if self.TDSW is None:

            with h5py.File(h5f, 'r') as file:

                # r = load_element_old(file, level_to_down_prefix(level) + phase)
                # TODO: minor resolve n,f,a
                # self.provider = DownProvider2( self.mesh_provider.n, self.TDSI, self.TDSW)

                self.TDSW, self.TDSI = su.batch_downsample_matrix(self.mesh_provider.getAllV(),
                                                                  self.mesh_provider_d.getAllV(), k=10)


        return self.TDSI[r,:,:], self.TDSW[r,:,:]
        '''

        return self.rDSI.get(r), self.rDSW.get(r)


class default_DownProvider:

    def print(self, s):
        print('DownProvider(): %s'%(s))

    def __init__(self, h5, phase, level, mesh_provider=None, mesh_provider_d=None):
        self.h5 = h5
        self.phase = phase
        self.level = level

        prefix = level_to_down_prefix(level) + phase

        def TR(variable):
            return TensorRegister(self.h5, prefix, variable)

        self.rDSI = TR('TDSI')
        self.rDSW = TR('TDSW')

        # note there is no first dim for batch!
        self.n = self.rDSW.shape[0]
        self.k = self.rDSW.shape[1]
        assert(self.k==self.rDSI.shape[1])

        # self.print('%s k=%d'%(prefix,self.k))

        return

    def get(self, r):

        return self.rDSI.get(r), self.rDSW.get(r)


class default_DownProvider_old:

    def __init__(self, n, TDSI, TDSW):

        self.n = n

        self.np_data = su.C()

        self.np_data.TDSI = TDSI
        self.np_data.TDSW = TDSW

        self.k = self.np_data.TDSI.shape[2]

        assert(self.k==self.np_data.TDSW.shape[2])

        return

    def get(self, r):

        TDSI_batch = self.np_data.TDSI[r, :, :]
        TDSW_batch = self.np_data.TDSI[r, :, :]

        return TDSI_batch, TDSW_batch


class default_LabelProvider:

    def print(self, s):
        print('LabelProvider(): %s'%(s))

    def __init__(self, h5, phase):
        self.h5 = h5
        self.phase = phase

        prefix = label_prefix() + phase

        def TR(variable):
            return TensorRegister(self.h5, prefix, variable)

        self.rY = TR('label')

        return

    def get(self, r):

        '''
        h5f = self.h5 + '.part0.h5'
        phase = self.phase

        if self.TY is None:

            with h5py.File(h5f, 'r') as file:

                if ('label_' + phase) in file:
                    print('Label loaded for phase %s.'%(phase))
                    Y = np.asarray(load_label(file, 'label_' + phase))
                else:
                    print('Warning: the training file has no labels! Fattal error for classification task!')
                    # TODO: minor: resolve this.
                    # Y = np.asarray(-np.ones([self.mesh.num_data, 1]))
                self.TY = Y[:, np.newaxis]  #  [batch,] -> [batch,1]

        return self.TY[r]
        '''

        return self.rY.get(r)


class default_LabelProvider2:

    def __init__(self, TY):

        self.np_data = su.C()

        self.np_data.TY = TY

        return

    def get(self, r):

        TY_batch = self.np_data.TY[r]

        return TY_batch


def default_MeshStructMetaReader(filename, mesh_level, phase):
    h5f = filename + '.part0.h5'

    with h5py.File(h5f, 'r') as file:

        if mesh_level is None:
            level = file.get('level').value
        else:
            level = mesh_level

        level_in_h5py = -1
        MAX_ALLOWED_LEVEL = 10
        for i in range(MAX_ALLOWED_LEVEL):
            prefix = level_to_mesh_prefix(i) + phase
            if prefix in file:
                level_in_h5py = i
        if level_in_h5py < level:
            print(
                'Warning: given level=%d, actual level=%d, additional levels not used.' % (level, level_in_h5py))
            # self.level = level_in_h5py

        num_data = file.get('total_num_data_' + phase).value

    return level, num_data


###################################################

def simple_create_mesh_tf(batch, provider):

    mesh = su.C()

    n = provider.n
    f = provider.f
    ta = provider.ta
    tb = provider.tb

    mesh.V = tf.placeholder(tf.float32, shape=[batch, n, 3], name='V')
    mesh.F = tf.placeholder(tf.int32, shape=[batch,f,3], name='F')

    mesh.A = tf.placeholder(tf.float32, shape=[batch,n,ta], name='A')
    mesh.b = tf.placeholder(tf.float32, shape=[batch,tb], name='b')

    return mesh


def simple_make_mesh_feed_list(tf_data, provider, out):

    feed_list = {
        tf_data.V: out['V'],
        tf_data.F: out['F'],
        tf_data.A: out['A'],
        tf_data.b: out['b']
    }
    return feed_list


class simple_MeshBatch:
    def __init__(self, provider, out):

        self.TV = out['V']
        self.TF = out['F']
        self.TA = out['A']
        self.Tb = out['b']


class simple_MeshProvider:
    # n, f, a, ta, tb, use_sparse, get, getV
    def print(self, s):
        print('MeshProvider(): %s'%(s))

    def __init__(self, h5, phase, level, max_adj):
        self.h5 = h5
        self.phase = phase
        self.level = level

        prefix = level_to_mesh_prefix(level) + phase

        def TR(variable):
            return TensorRegister(self.h5, prefix, variable)

        self.rV = TR('TV')
        self.rF = TR('TF')
        self.rA = TR('TA')
        self.rb = TR('Tb')
        # # note there is no first dim for batch!
        self.n = self.rV.shape[0]
        self.f = self.rF.shape[0]

        self.ta = self.rA.shape[1]
        self.tb = self.rb.shape[0]


        self.print('%s n=%d, f=%d, ta=%d, tb=%d'%(prefix,self.n,self.f,self.ta,self.tb))

        return

    def get(self, r, augmentor=None):

        if augmentor==None:
            augmentor = Augmentor()

        return {'V':augmentor.apply(modify_padding(self.rV.get(r))),
                    'F': self.rF.get(r), 'A': self.rA.get(r), 'b':self.rb.get(r)}

    def getV(self, r):

        return modify_padding(self.rV.get(r))

###################################################

###################################################

###################################################

def sparse_dispatch(Js):

    # Js is a 3-nested list.

    total = sum([sum([len(Js[i][j]) for j in range(len(Js[i]))]) for i in range(len(Js))])

    TB = np.zeros([total])
    TR = np.zeros([total])
    TC = np.zeros([total])
    TV = np.zeros([total])

    index = 0

    for b in range(len(Js)):
        rJ, cJ, vJ = sparse_stencil(Js[b])
        lb = rJ.shape[0]

        TB[index:index + lb] = b
        TR[index:index + lb] = rJ
        TC[index:index + lb] = cJ
        TV[index:index + lb] = vJ

        index = index + lb

    return TB, TR, TC, TV


def create_mesh_attr(batch, n, f, va, fa):

    mesh_attr = su.C()
    mesh_attr.YVA = tf.placeholder(tf.float32, shape=[batch,n,va], name='YVA')
    mesh_attr.YFA = tf.placeholder(tf.float32, shape=[batch,f,fa], name='YFA')

    return mesh_attr


def make_mesh_attr_feed_list(tf_data, TYVA_batch, TTFA_batch):
    feed_list = {
        tf_data.YVA: TYVA_batch,
        tf_data.YFA: TTFA_batch
    }
    return feed_list


class MeshAttrBatch:

    def __init__(self, TVA, TFA):

        self.TVA = TVA
        self.TFA = TFA

    def make_feed_list(self, mesh_attr):

        return make_mesh_attr_feed_list(mesh_attr.tf_data, self.TVA, self.TFA)


import h5py


def get_item(f, name, variable, idx):
    return np.asarray(f.get(name + '/%06d/%s' % (idx, variable)))


class TensorRegister:

    def __init__(self, h5, prefix, variable):

        self.h5 = h5
        self.prefix = prefix
        self.variable = variable


        h5f = h5 + '.part0.h5'

        with h5py.File(h5f, 'r') as file:

            def get_first_local(variable):
                if prefix not in file:
                    print('Error: TensorRegister() cannot find'+variable+'in'+prefix)
                return get_item(file, prefix, variable, 0)

            T0 = get_first_local(variable)

            self.shape = T0.shape
            # print('TensorRegister(): '+variable+' registered with shape=',self.shape)

            '''
            This are code for possible prefetching from hard disk.
            if 'total_num_fold' in file:
                self.num_fold = file.get('total_num_fold')
            else:
                print('Warning: no total_num_fold provided.')
                self.num_fold = 1
            '''

        '''
        self.fold_ranges = [[-1,-1] for i in range(self.num_fold)]

        for i in range(self.num_fold):
            with h5py.File(h5f, 'r') as file:
                self.fold_ranges[i][0] = file.get(prefix + 'fold_range_lo')
                self.fold_ranges[i][1] = file.get(prefix + 'fold_range_hi')

                if i > 0:
                    assert (self.fold_ranges[i][0]==self.fold_ranges[i-1][1])
        '''

    def get(self, r):

        prefix = self.prefix
        variable = self.variable
        h5f = self.h5 + '.part0.h5'

        T = []

        with h5py.File(h5f, 'r') as file:

            T = np.stack([get_item(file, prefix, variable, idx) for idx in r], axis=0)

        return T


class SparseTensor():

    def __init__(self, shape, Js):

        self.shape = shape
        self.Js = Js
        return

    def get_feed_dict(self, prefix=''):

        assert (len(self.Js) == self.shape[0])

        TB, TR, TC, TV = sparse_dispatch(self.Js)

        # return tf.SparseTensorValue(np.stack([TB,TR,TC], axis=0), TV, self.shape)
        return {prefix + 'b': TB,
                prefix + 'r': TR,
                prefix + 'c': TC,
                prefix + 'v': TV}

    def __init2__(self, shape, rows_batch, cols_batch, values_batch):

        assert (len(rows_batch)==len(cols_batch))
        assert (len(rows_batch)==len(values_batch))

        self.shape = [len(rows_batch), shape[0], shape[1]]

        self.rows_batch = rows_batch
        self.cols_batch = cols_batch
        self.values_batch = values_batch

        return


# This is currently unused
class SparseTensorRegister:

    def __init__(self, h5, prefix, variable):

        self.h5 = h5
        self.prefix = prefix
        self.variable = variable


        h5f = h5 + '.part0.h5'

        with h5py.File(h5f, 'r') as file:

            def get_first_local(variable):
                if prefix not in file:
                    print('Error: TensorRegister() cannot find'+variable+'in'+prefix)
                return get_item(file, prefix, variable, 0)

            # get_first_local(variable+'/rows')
            # get_first_local(variable + '/cols')
            self.shape = get_first_local(variable + '/shape')


    def get(self, r):

        prefix = self.prefix
        variable = self.variable
        h5f = self.h5 + '.part0.h5'

        T  = []
        with h5py.File(h5f, 'r') as file:

            # assume get_first_local(variable + '/shape') == self.shape

            T = SparseTensor(self.shape,
                        [get_item(file, prefix, variable + '/rows', idx) for idx in r],
                        [get_item(file, prefix, variable + '/cols', idx) for idx in r],
                        [get_item(file, prefix, variable + '/values', idx) for idx in r])

        return T


def level_to_mesh_name(level):
    if level==0:
        return 'mesh'
    elif level==1:
        return 'mesh_d'
    else:
        return 'mesh_d%d'%(level)


def level_to_mesh_prefix(level):
    return level_to_mesh_name(level)+'_'


def remove_pads(TF):

    return [TF[b, np.where(TF[b,:,0] > -1),:][0] for b in range(TF.shape[0])] # I do not totally understand why I need [0] but it should be there to work.


def sparse_stencil(J):

    total = sum([len(J[i]) for i in range(len(J))])

    rows = np.zeros([total], dtype='int32')
    cols = np.zeros([total], dtype='int32')
    values = np.zeros([total], dtype='float64')

    index = 0
    for i in range(len(J)):
        for j in range(len(J[i])):

            rows[index] = i
            cols[index] = j
            values[index] = J[i][j]

            index = index + 1

    return rows, cols, values


def faces_to_adjacencies(TF, shape):
    batch = TF.shape[0]
    n, a = shape[0], shape[1]

    Fs = remove_pads(TF)
    # list of np matrices.

    print('faces_to_adjacencies: Fs', len(Fs), Fs[0].shape)

    Js, Ks = su.meshes_adjacenies_compact(n, Fs)

    J_list = SparseTensor([batch, n, a], Js).get_feed_dict(prefix='J/')
    K_list = SparseTensor([batch, n, a], Ks).get_feed_dict(prefix='K/')

    return J_list, K_list


def modify_padding(T):
    MT = T # np.zeros_like(T)
    for b in range(T.shape[0]):
        t = T[b]
        padded_indices = np.where((t[:,0]<-999) & (t[:,0]>-1001)) # cannot use and
        # print('padded_indices:',padded_indices)
        MT[b, padded_indices, :] = np.zeros_like(t[padded_indices, :])

    return MT


def MeshGeneric(G_MESH_BATCH, G_TF_FUN, G_MAKE_FEED_LIST):
    class MeshLocal:

        def __init_old2__(self, Vs, Fs, As, bs, Js, Ks, n, f, a, names=None):
            # a is the maximum adjacent vertices per vertex.

            self.np_data = su.C()

            self.np_data.TV, self.np_data.TF, self.np_data.TJ, self.np_data.TK, self.np_data.TA, self.np_data.Tb = \
                data_tensors( Vs, Fs, As, bs, Js, Ks, n=n, f=f, a=a, names=names)

            self.ta = self.np_data.TA.shape[2]
            self.tb = self.np_data.Tb.shape[1]

            self.num_data = self.np_data.TV.shape[0]

            self.n = n
            self.f = f
            self.a = a

        def __init_old__(self, Vs, Fs, As, bs, Js, Ks,  n, f, a, names=None):
        # def __init__(self, Vs, Fs, As, bs, n, f, a, names=None):
            # a is the maximum adjacent vertices per vertex.

            self.np_data = su.C()

            self.np_data.TV, self.np_data.TF, self.np_data.TJ, self.np_data.TK, self.np_data.TA, self.np_data.Tb = \
                data_tensors_old( Vs, Fs, As, bs, n=n, f=f, a=a, names=names)

            self.ta = self.np_data.TA.shape[2]
            self.tb = self.np_data.Tb.shape[1]

            self.num_data = self.np_data.TV.shape[0]

            self.n = n
            self.f = f
            self.a = a

        def __init__(self, provider, name):

            self.tf_data = None # To be assigned later.
            self.provider = provider

            self.num_data = None
            self.num_batch = None
            self.name = name

        def init_tf_data(self, batch, external_tf_data=[]):

            self.batch = batch

            if not external_tf_data: # when tf_data==[]
                self.tf_data = G_TF_FUN(self.batch, self.provider)
            else:
                self.tf_data = external_tf_data

        def make_batch(self, batch_index, augmentor=None):

            ''' TODO:
            self.num_data = self.provider

            # assert(np.mod(self.num_data,batch)==0)
            if np.mod(self.num_data,batch)!=0:
                print('Warning! data counts %d cannot be divided by batch_size %d, last data discarded (kept but unused).'%(self.num_data, batch ))

            self.num_batch = self.num_data // batch
            '''

            r = np.arange(self.batch * batch_index, self.batch * (batch_index + 1))

            out = self.provider.get(r, augmentor)

            return G_MESH_BATCH(self.provider, out), G_MAKE_FEED_LIST(self.tf_data, self.provider, out)

    return MeshLocal

#Mesh = MeshGeneric(default_MeshBatch, default_create_mesh_tf, default_make_mesh_feed_list)
#MeshProvider = default_MeshProvider


class MeshAttrProvider2:

    def __init__(self, TVAs, TFAs, n, f):

        self.np_data = su.C()

        self.np_data.TYVA, self.np_data.TYFA = \
            data_attr_tensors( TVAs, TFAs, n=n, f=f)

        self.va = self.np_data.TYVA.shape[2]
        self.fa = self.np_data.TYFA.shape[2]

        self.num_data = self.np_data.TYVA.shape[0]

        self.n = n
        self.f = f

    def get(self, r):

        TYVA_batch = self.np_data.TYVA[r, :, :]
        TYFA_batch = self.np_data.TYFA[r, :, :]

        return TYVA_batch, TYFA_batch


class MeshAttr:

    def __init__(self, provider):

        self.provider = provider

    def init_tf_data(self, batch, external_tf_data=[]):

        self.batch = batch

        assert(np.mod(self.num_data,batch)==0)

        self.num_batch = self.num_data // batch

        if not external_tf_data: # when tf_data==[]
            self.tf_data = create_mesh_attr(self.batch, self.provider.n, self.provider.f, self.provider.va, self.provider.fa)
        else:
            self.tf_data = external_tf_data

    def make_batch(self, batch_index):

        r = np.arange(self.batch * batch_index, self.batch * (batch_index + 1))

        TYVA_batch, TYFA_batch = self.provider.get(r)

        return MeshAttrBatch( TYVA_batch, TYFA_batch), make_mesh_attr_feed_list( self.tf_data, TYVA_batch, TYFA_batch)


def level_to_down_name(level):
    assert(level>=1)
    return 'ds%d%d'%(level-1,level) # e.g. ds01


def level_to_down_prefix(level):
    return  level_to_down_name(level)+'_'


class Down:

    def __init__(self, provider):

        self.provider = provider

    def init_tf_data(self, batch, external_tf_data=[]):

        self.batch = batch

        #assert(np.mod(self.num_data,batch)==0)

        #self.num_batch = self.num_data // batch
        if not external_tf_data:
            self.tf_data = su.C()

            self.tf_data.index = tf.placeholder(tf.int32, shape=[batch, self.provider.n, self.provider.k])
            self.tf_data.weight = tf.placeholder(tf.float32, shape=[batch, self.provider.n, self.provider.k])

        else:
            self.tf_data = external_tf_data

    def make_batch(self, batch_index):

        r = np.arange(self.batch * batch_index, self.batch * (batch_index + 1))

        TDSI_batch, TDSW_batch = self.provider.get(r)

        return { self.tf_data.index: TDSI_batch, self.tf_data.weight: TDSW_batch}


class LabelBatch:

    def __init__(self, TY):

        self.TY = TY

    def make_feed_list(self, label):

        raise NameError('Not Implemented yet!')

        return


def label_name():
    return 'label'


def label_prefix():
    return label_name()+'_'


class Label:

    def __init__(self, provider):

        self.provider = provider

    def init_tf_data(self, batch, external_tf_data=[]):

        self.batch = batch

        #assert(np.mod(self.num_data,batch)==0)

        #self.num_batch = self.num_data // batch

        if not external_tf_data:
            self.tf_data = su.C()

            self.tf_data.y_label = tf.placeholder(tf.int32, [batch, 1])
        else:
            self.tf_data = external_tf_data

    def make_batch(self, batch_index):

        r = np.arange(self.batch * batch_index, self.batch * (batch_index + 1))

        TY_batch = self.provider.get(r)

        TY_batch = TY_batch[:, np.newaxis]  # [batch,] -> [batch,1]

        return LabelBatch(TY_batch), { self.tf_data.y_label: TY_batch}


def load_label(f, name):

    r = [np.asarray(f.get(name + '/%06d/label' % (i))) for i in range(len(f[name]))]

    return r


def load_element(f, name):
    r = su.C()

    r.Vs = [np.asarray(f.get(name + '/%06d/V' % (i))) for i in range(len(f[name]))]
    r.Fs = [np.asarray(f.get(name + '/%06d/F' % (i))) for i in range(len(f[name]))]
    r.As = [np.asarray(f.get(name + '/%06d/A' % (i))) for i in range(len(f[name]))]
    r.bs = [np.asarray(f.get(name + '/%06d/b' % (i))) for i in range(len(f[name]))]
    r.Js = [np.asarray(f.get(name + '/%06d/J' % (i))) for i in range(len(f[name]))]
    r.Ks = [np.asarray(f.get(name + '/%06d/K' % (i))) for i in range(len(f[name]))]
    r.names = [(f.get(name + '/%06d/name' % (i))).value for i in range(len(f[name]))]

    for i in range(len(r.Vs)):
        assert (r.Vs[i].shape[0] > 0 and r.Vs[i].shape[1] == 3)
        assert (r.Fs[i].shape[0] > 0 and r.Fs[i].shape[1] == 3)
        assert (r.As[i].shape[0] == r.Vs[i].shape[0] and r.As[i].shape[1] > 0)
        assert (r.bs[i].shape[0] > 0)
        print(r.names[i])

    return r


def load_element_old(f, name):
    r = su.C()

    r.Vs = [np.asarray(f.get(name + '/%06d/V' % (i))) for i in range(len(f[name]))]
    r.Fs = [np.asarray(f.get(name + '/%06d/F' % (i))) for i in range(len(f[name]))]
    r.As = [np.asarray(f.get(name + '/%06d/A' % (i))) for i in range(len(f[name]))]
    r.bs = [np.asarray(f.get(name + '/%06d/b' % (i))) for i in range(len(f[name]))]
    r.names = [(f.get(name + '/%06d/name' % (i))).value for i in range(len(f[name]))]

    #for i in range(len(r.Vs)):
        #if (r.bs[i].size<=0):
    #    print('Warning: r.bs[%d] is empty, auto resizing.')
    #    r.bs[i] = np.zeros([1])

    for i in range(len(r.Vs)):
        assert (r.Vs[i].shape[0] > 0 and r.Vs[i].shape[1] == 3)
        assert (r.Fs[i].shape[0] > 0 and r.Fs[i].shape[1] == 3)
        assert (r.As[i].shape[0] == r.Vs[i].shape[0] and r.As[i].shape[1] > 0)
        assert (r.bs[i].shape[0] > 0)
        print(r.names[i])

    return r


def make_dim2(IA):
    if IA.ndim >=2:
        OA = IA
    elif IA.ndim<1:
        OA = np.zeros([0,0])
    elif IA.shape[0]>0:
        OA = IA[:,np.newaxis]
    else:
        OA = np.zeros([0, 0])
    return OA


def load_element_attr(f, name):
    r = su.C()
    # TODO: maybe see if the enties exist or not.
    r.YVAs = [ make_dim2( np.asarray(f.get(name + '/%06d/YVA' % (i))) ) for i in range(len(f[name]))]
    r.YFAs = [ make_dim2( np.asarray(f.get(name + '/%06d/YFA' % (i))) ) for i in range(len(f[name]))]

    return r


class Augmentor:

    def __init__(self):
        return
    #def start(self):
    #    return
    def apply(self, data):
        return data

class TestNoiser:

    def __init__(self):
        return
    #def start(self):
    #    return
    def apply(self, data):
        return data


'''
import os
def load_mesh_struct(dataset, phase, prefetch=False):
    if prefetch and \
            ( os.path.exists(dataset + '.' + phase + '.shelve.dir') or # win
            os.path.exists(dataset + '.' + phase + '.shelve.db') ): # linux
        mesh_struct = ss.MeshStruct.from_shelve(dataset + '.' + phase + '.shelve')
    else:
        mesh_struct = ss.MeshStruct(dataset, phase, n, f, a)
        if prefetch:
            try:
                mesh_struct.save_tf(dataset + '.' + phase + '.shelve')
            except:
                print('Fail to shelve the tf data')
    return mesh_struct
'''



def MeshStructGeneric(G_MESH, G_MESH_STRUCT_META_READER, G_MESH_PROVIDER, G_DOWN_PROVIDER, G_LABEL_PROVIDER):

    class MeshStructLocal:

        def print(self, s):
            print('MeshStruct(): %s'%(s))

        def get_num_data(self):
            # this return the total number of data in the dataset, not just in one fold.
            return self.num_data

        def __init__(self, h5, phase, mesh_level=None, max_adj=None):

            assert(phase=='train' or phase=='test')

            self.name = phase
            #self.use_sparse = use_sparse


            self.level, self.num_data = G_MESH_STRUCT_META_READER(h5, mesh_level, phase)

            self.print('Level set to %d' % (self.level))
            self.print('Total num data is %d' % (self.num_data))


            level = self.level

            if True:

                self.mesh = None
                self.mesh_d = None
                self.mesh_d2 = None
                self.mesh_d3 = None
                self.mesh_d4 = None

                self.ds01 = None
                self.ds12 = None
                self.ds23 = None
                self.ds34 = None

                self.mesh = G_MESH(G_MESH_PROVIDER(h5, phase, level=0, max_adj=max_adj), 'mesh')
                if level >= 1:
                    self.mesh_d = G_MESH(G_MESH_PROVIDER(h5, phase, level=1, max_adj=max_adj), 'mesh_d')
                if level >= 2:
                    self.mesh_d2 = G_MESH(G_MESH_PROVIDER(h5, phase, level=2, max_adj=max_adj), 'mesh_d2')
                if level >= 3:
                    self.mesh_d3 = G_MESH(G_MESH_PROVIDER(h5, phase, level=3, max_adj=max_adj), 'mesh_d3')
                if level >= 4:
                    self.mesh_d4 = G_MESH(G_MESH_PROVIDER(h5, phase, level=4, max_adj=max_adj), 'mesh_d4')

                # TODO: fixed this later.
                # if ('mesh_attr_' + phase) in file:  # TODO: do this only if the entries exist.
                #    r_attr = load_element_attr(file, 'mesh_attr_' + phase)
                #    self.mesh_attr = MeshAttr(MeshAttrProvider(r_attr.YVAs, r_attr.YFAs, n, f))

                if G_DOWN_PROVIDER is not None:
                    if level >= 1:
                        # self.ds01 = Down(n // 4, self.TDSI, self.TDSW)
                        self.ds01 = Down(
                            G_DOWN_PROVIDER(h5, phase, 1, self.mesh.provider, self.mesh_d.provider))
                    if level >= 2:
                        # self.ds12 = Down(n // 16, self.TDSI2, self.TDSW2)
                        self.ds12 = Down(
                            G_DOWN_PROVIDER(h5, phase, 2, self.mesh_d.provider, self.mesh_d2.provider))
                    if level >= 3:
                        # self.ds23 = Down(n // 64, self.TDSI3, self.TDSW3)
                        self.ds23 = Down(
                            G_DOWN_PROVIDER(h5, phase, 3, self.mesh_d2.provider, self.mesh_d3.provider))
                    if level >= 4:
                        self.ds34 = Down(
                            G_DOWN_PROVIDER(h5, phase, 4, self.mesh_d3.provider, self.mesh_d4.provider))

                if G_LABEL_PROVIDER is not None:
                    self.label = Label(G_LABEL_PROVIDER(h5, phase))

        def __init_old__(self, h5f, phase, n, f, a):

            self.name = phase

            with h5py.File(h5f, 'r') as file:

                level = 0
                if ('mesh_d_' + phase) in file: # len(file['mesh_d_' + phase]) > 0:
                    level = 1
                if ('mesh_d2_' + phase) in file: #len(file['mesh_d2_' + phase]) > 0:
                    level = 2
                if ('mesh_d3_' + phase) in file: #len(file['mesh_d3_' + phase]) > 0:
                    level = 3
                if ('mesh_d4_' + phase) in file: #len(file['mesh_d4_' + phase]) > 0:
                    level = 4

                self.level = level

                # print('Level set to %d'%(level))


                # self.TY = np.asarray(file.get('label_' + phase, default=-np.ones([self.mesh.num_data,1])))  # this is already 1000 by 1
                # This old format works for only the mnist data.

                if ('label_' + phase) in file:
                    Y = np.asarray(load_label(file, 'label_' + phase))
                else:
                    print('Warning: the training file has no labels! Fattal error for classification task!')
                    Y = np.asarray(-np.ones([self.mesh.num_data, 1]))
                self.TY = Y[:, np.newaxis]  #  [n,] -> [n,1]

                # for i in range(100):
                #     print('TY[%4d]=%d'%(i,self.TY[i]))

                r = load_element(file, 'mesh_' + phase)
                if level >= 1:
                    r_d = load_element(file, 'mesh_d_' + phase)
                if level >= 2:
                    r_d2 = load_element(file, 'mesh_d2_' + phase)
                if level >= 3:
                    r_d3 = load_element(file, 'mesh_d3_' + phase)
                if level >= 4:
                    r_d4 = load_element(file, 'mesh_d4_' + phase)

                if True:

                    self.mesh = Mesh( MeshProvider2(r.Vs, r.Fs, r.As, r.bs, r.Js, r.Ks, n, f, a, names=r.names) )
                    if level >= 1:
                        self.mesh_d = Mesh( MeshProvider2(r_d.Vs, r_d.Fs, r_d.As, r_d.bs, r_d.Js, r_d.Ks, n, f, a, names=r_d.names) )
                        # self.mesh_d = Mesh(r_d.Vs, r_d.Fs, r_d.As, r_d.bs, n // 4, f // 4, a, names=r_d.names)
                    if level >= 2:
                        self.mesh_d2 = Mesh( MeshProvider2(r_d2.Vs, r_d2.Fs, r_d2.As, r_d2.bs, r_d2.Js, r_d2.Ks, n // 4, f // 4, a, names=r_d2.names) )
                        # self.mesh_d2 = Mesh(r_d2.Vs, r_d2.Fs, r_d2.As, r_d2.bs, n // 16, f // 16, a, names=r_d2.names)
                    if level >= 3:
                        self.mesh_d3 = Mesh( MeshProvider2(r_d3.Vs, r_d3.Fs, r_d3.As, r_d3.bs, r_d3.Js, r_d3.Ks, n // 16, f // 16, a, names=r_d3.names) )
                        # self.mesh_d3 = Mesh(r_d3.Vs, r_d3.Fs, r_d3.As, r_d3.bs, n // 64, f // 64, a, names=r_d3.names)
                    if level >= 4:
                        self.mesh_d4 = Mesh( MeshProvider2(r_d4.Vs, r_d4.Fs, r_d4.As, r_d4.bs, r_d4.Js, r_d4.Ks, n // 256, f // 256, a, names=r_d4.names) )

                    if ('mesh_attr_' + phase) in file: # TODO: do this only if the entries exist.
                        r_attr = load_element_attr(file, 'mesh_attr_' + phase)
                        self.mesh_attr = MeshAttr( MeshAttrProvider2( r_attr.YVAs, r_attr.YFAs, n, f) )

                    # this step can be slow, consider precompute this step if possible.

                    if level >= 1:
                        self.TDSW, self.TDSI = su.batch_downsample_matrix(self.mesh.provider.getV(), self.mesh_d.provider.getV(), k=10)
                    if level >= 2:
                        self.TDSW2, self.TDSI2 = su.batch_downsample_matrix(self.mesh_d.provider.getV(), self.mesh_d2.provider.getV(), k=10)
                    if level >= 3:
                        self.TDSW3, self.TDSI3 = su.batch_downsample_matrix(self.mesh_d2.provider.getV(), self.mesh_d3.provider.getV(), k=10)
                    if level >= 4:
                        self.TDSW4, self.TDSI4 = su.batch_downsample_matrix(self.mesh_d3.provider.getV(), self.mesh_d4.provider.getV(), k=10)

                    if level >= 1:
                        # self.ds01 = Down(n // 4, self.TDSI, self.TDSW)
                        self.ds01 = Down( DownProvider2( n, self.TDSI, self.TDSW) )
                    if level >= 2:
                        # self.ds12 = Down(n // 16, self.TDSI2, self.TDSW2)
                        self.ds12 = Down( DownProvider2( n // 4, self.TDSI2, self.TDSW2) )
                    if level >= 3:
                        # self.ds23 = Down(n // 64, self.TDSI3, self.TDSW3)
                        self.ds23 = Down( DownProvider2( n // 16, self.TDSI3, self.TDSW3) )
                    if level >= 4:
                        self.ds34 = Down( DownProvider2( n // 256, self.TDSI4, self.TDSW4) )
                    self.label = Label( LabelProvider2( self.TY) )

        def print_tf(self, filename):
            for key,value in zip(self.__dict__.keys(), self.__dict__.values()):
                print(key, ": ", value)

        def save_tf(self, filename):
            import shelve

            #filename = '/tmp/shelve.out'
            my_shelf = shelve.open(filename, 'n')  # 'n' for new

            for key,value in zip(self.__dict__.keys(), self.__dict__.values()):
                # print(key, ": ", value)
                try:
                    my_shelf[key] = value
                except TypeError:
                    #
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    #
                    print('ERROR shelving: {0}'.format(key))
            my_shelf.close()

        @classmethod
        def from_shelve(cls, filename):
            # https://stackoverflow.com/questions/141545/how-to-overload-init-method-based-on-argument-type
            # "Initialize MyData from a dict's items"

            print("Loading from shelve...", filename)

            tf_dict = cls.load_tf(filename)
            #return cls(tf_dict.items())

            obj = cls.__new__(cls)

            for key in tf_dict:
                setattr(obj, key, tf_dict[key])

            return obj

        @classmethod
        def load_tf(cls, filename):
            import shelve

            my_shelf2 = shelve.open(filename, 'r')

            #for key in my_shelf2:
            #    print(key, my_shelf2[key])

            tf_dict = {k: my_shelf2[k] for k in my_shelf2}

            return tf_dict

        def __init__mnist__(self, mat, name):

            self.name = name

            self.mesh = ss.Mesh(mat['Vs_' + self.name], mat['Fs_' + self.name], n, f, a)
            self.mesh_d = ss.Mesh(mat['Vs_' + self.name + '_d'], mat['Fs_' + self.name + '_d'], n // 4, f // 4, a)
            self.mesh_d2 = ss.Mesh(mat['Vs_' + self.name + '_d2'], mat['Fs_' + self.name + '_d2'], n // 16, f // 16, a)

            self.TY = mat['Ys_' + self.name][0][:, np.newaxis]  # remove the first redundant axis

            self.DSI, self.DSW = su.batch_downsample_matrix(self.mesh_d.num_data, 28, 28)
            self.DSI2, self.DSW2 = su.batch_downsample_matrix(self.mesh_d.num_data, 14, 14)

            self.TDSI = su.meshes_index_attributes_to_tensor(self.DSI, n // 4)
            self.TDSW = su.meshes_attributes_to_tensor(self.DSW, n // 4)

            self.TDSI2 = su.meshes_index_attributes_to_tensor(self.DSI2, n // 16)
            self.TDSW2 = su.meshes_attributes_to_tensor(self.DSW2, n // 16)

            self.ds01 = ss.Down(n // 4, self.TDSI, self.TDSW)
            self.ds12 = ss.Down(n // 16, self.TDSI2, self.TDSW2)

            self.label = ss.Label(self.TY)

            print("Dim: %d", self.TDSI.shape[0])

        def __init_new__(self, mat, name):

            self.name = name

            self.mesh = ss.Mesh(mat['Vs_' + self.name], mat['Fs_' + self.name].astype(np.int32), n, f, a)
            self.mesh_d = ss.Mesh(mat['Vs_' + self.name + '_d'], mat['Fs_' + self.name + '_d'].astype(np.int32), n // 4,
                                  f // 4, a)
            self.mesh_d2 = ss.Mesh(mat['Vs_' + self.name + '_d2'], mat['Fs_' + self.name + '_d2'].astype(np.int32), n // 16,
                                   f // 16, a)

            self.TY = mat['Ys_' + self.name][0][:, np.newaxis]  # remove the first redundant axis

            self.TDSW = mat['TDSW_' + self.name]
            self.TDSI = mat['TDSI_' + self.name].astype(np.int32)

            self.TDSW2 = mat['TDSW2_' + self.name]
            self.TDSI2 = mat['TDSI2_' + self.name].astype(np.int32)

            self.ds01 = ss.Down(n // 4, self.TDSI, self.TDSW)
            self.ds12 = ss.Down(n // 16, self.TDSI2, self.TDSW2)

            self.label = ss.Label(self.TY)

        def init_tf(self, batch_size, external_minst_tf_data=[]):

            self.batch = batch_size

            if not external_minst_tf_data:  # when tf_data==[]
                self.mesh.init_tf_data(self.batch)
                if self.level >= 1:
                    self.mesh_d.init_tf_data(self.batch)
                if self.level >= 2:
                    self.mesh_d2.init_tf_data(self.batch)
                if self.level >= 3:
                    self.mesh_d3.init_tf_data(self.batch)
                if self.level >= 4:
                    self.mesh_d4.init_tf_data(self.batch)

                if G_DOWN_PROVIDER is not None:
                    if self.level >= 1:
                        self.ds01.init_tf_data(self.batch)
                    if self.level >= 2:
                        self.ds12.init_tf_data(self.batch)
                    if self.level >= 3:
                        self.ds23.init_tf_data(self.batch)
                    if self.level >= 4:
                        self.ds34.init_tf_data(self.batch)

                if hasattr(self, 'mesh_attr'):
                    self.mesh_attr.init_tf_data(self.batch)

                if G_LABEL_PROVIDER is not None:
                    self.label.init_tf_data(self.batch)
            else:
                self.mesh.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh.tf_data)
                if self.level >= 1:
                    self.mesh_d.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh_d.tf_data)
                if self.level >= 2:
                    self.mesh_d2.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh_d2.tf_data)
                if self.level >= 3:
                    self.mesh_d3.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh_d3.tf_data)
                if self.level >= 4:
                    self.mesh_d4.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh_d4.tf_data)
                if G_DOWN_PROVIDER is not None:
                    if self.level >= 1:
                        self.ds01.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.ds01.tf_data)
                    if self.level >= 2:
                        self.ds12.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.ds12.tf_data)
                    if self.level >= 3:
                        self.ds23.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.ds23.tf_data)
                    if self.level >= 4:
                        self.ds34.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.ds34.tf_data)
                if hasattr(self, 'mesh_attr'):
                    self.mesh_attr.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.mesh_attr.tf_data)
                if G_LABEL_PROVIDER is not None:
                    self.label.init_tf_data(self.batch, external_tf_data=external_minst_tf_data.label.tf_data)

        def make_batch(self, batch_index, augmentor=Augmentor()):

            s = batch_index
            # s: index of this batch.

            feed = {}

            # augmentor.start()

            if G_DOWN_PROVIDER is not None:
                if self.level >= 1:
                    ds01_list = self.ds01.make_batch(s)
                    feed.update(ds01_list)
                if self.level >= 2:
                    ds12_list = self.ds12.make_batch(s)
                    feed.update(ds12_list)
                if self.level >= 3:
                    ds23_list = self.ds23.make_batch(s)
                    feed.update(ds23_list)
                if self.level >= 4:
                    ds34_list = self.ds34.make_batch(s)
                    feed.update(ds34_list)

            if hasattr(self, 'mesh_attr'):
                mesh_attr_batch, mesh_attr_list = self.mesh_attr.make_batch(s)
                feed.update(mesh_attr_list)

            mesh_batch, mesh_list = self.mesh.make_batch(s, augmentor=augmentor)
            feed.update(mesh_list)
            if self.level >= 1:
                mesh_d_batch, mesh_d_list = self.mesh_d.make_batch(s, augmentor=augmentor)
                feed.update(mesh_d_list)
            if self.level >= 2:
                mesh_d2_batch, mesh_d2_list = self.mesh_d2.make_batch(s, augmentor=augmentor)
                feed.update(mesh_d2_list)
            if self.level >= 3:
                mesh_d3_batch, mesh_d3_list = self.mesh_d3.make_batch(s, augmentor=augmentor)
                feed.update(mesh_d3_list)
            if self.level >= 4:
                mesh_d4_batch, mesh_d4_list = self.mesh_d4.make_batch(s, augmentor=augmentor)
                feed.update(mesh_d4_list)

            if G_LABEL_PROVIDER is not None:
                label_batch, label_list = self.label.make_batch(s)
                feed.update(label_list)

            return feed

    return MeshStructLocal



#Mesh = MeshGeneric(default_MeshBatch, default_create_mesh_tf, default_make_mesh_feed_list)
#MeshProvider = default_MeshProvider
#MeshStruct = MeshStructGeneric(Mesh, default_MeshStructMetaReader, MeshProvider, default_DownProvider, default_LabelProvider)


Mesh = MeshGeneric(simple_MeshBatch, simple_create_mesh_tf, simple_make_mesh_feed_list)
MeshProvider = simple_MeshProvider
MeshStruct = MeshStructGeneric(Mesh, default_MeshStructMetaReader, MeshProvider, default_DownProvider, default_LabelProvider)



def name_function(dataset_name,i):
    return dataset_name + '%d' % (i) + '.h5'

#class MeshStructBatch(MeshStruct):


'''
class MeshStructDataset:

    def __init__(self, dataset_name, phase, n, f, a):

        self.num_data = None
        self.num_fold = None

        import os

        self.num_fold = 0

        for i in range(1e8):
            if os.path.isfile( name_function(dataset_name,i) ):
                self.num_fold = i
        print('The dataset has %d folds.'%(self.num_fold))
        assert(self.num_fold>=1)

        self.fold_been_read = [False for i in range(self.num_fold)]
        self.fold_ranges = [[0,-1] for i in range(self.num_fold)] # fold_ranges[i] is valid iff fold_been_read[i]==True
        self.fold_filenames = [name_function(dataset_name,i) for i in range(self.num_fold)]

        self.fold_structs = [[] for i in range(self.num_fold)]

        self.current_fold_idx = 0



        for i in range(self.num_fold):
            with h5py.File( self.fold_filenames, 'r') as file:


    # def init_tf(self, batch_size, external_minst_tf_data=[]):
'''


def mref(mesh_struct, level):
    return getattr(mesh_struct, level_to_mesh_name(level)).provider


def dref(mesh_struct, level):
    return getattr(mesh_struct, level_to_down_name(level)).provider


def lref(mesh_struct):
    return getattr(mesh_struct, label_name()).provider


def cls_loss_base(label, logits, n_class):

    one_hot_y = tf.one_hot(label, depth=n_class)

    if (tf.__version__) == '1.4.0':
        # this is a hack on my win machine.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
    #
    loss = tf.reduce_mean(cross_entropy)
    return loss


def cls_loss(mesh_struct, logits, n_class):
    label = mesh_struct.label.tf_data.y_label[:, 0]
    return cls_loss_base(label, logits, n_class)

'''
def cls_loss_50(mnist, logits):
    n_class = 50
    one_hot_y = tf.one_hot(mnist.label.tf_data.y_label[:, 0], depth=n_class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    return loss
'''


def cls_loss_40(mesh_struct, logits):
    return cls_loss(mesh_struct, logits, 40)

''' old
def att_loss(mnist, att):
    loss = tf.reduce_mean(att-mnist.label.tf_data.y_label)
    return loss
'''


def attr_loss(mesh_struct, attr_pred, attr_truth):

    return tf.reduce_mean(tf.linalg.norm(attr_pred - attr_truth, axis=0, keep_dims=True))


def cls_accuracy(mnist, logits):
    correct_prediction = tf.reduce_sum(tf.cast(
            tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), mnist.label.tf_data.y_label[:, 0])
        , tf.int32))
    return tf.squeeze(correct_prediction).eval() * 1.0 / mnist.batch


class AttrTrainOp:

    def __init__(self, n_train, n_val, batch, out_train, log_print):

        self.n_train = n_train
        self.n_val = n_val
        self.batch = batch
        self.att_train = out_train.logits # todo: optional consider change this naming.
        self.log_print = log_print

        [_, n, self.c_att] = self.att_train.get_shape().as_list()

        self.att_table_train = np.zeros([self.n_train, n, self.c_att])
        self.att_table_val   = np.zeros([self.n_val, n, self.c_att])

        self.attr_loss_table_train = np.zeros([self.n_train//self.batch])
        self.attr_loss_table_val   = np.zeros([self.n_val//self.batch])

    def update(self, batch_idx, feed_train, losses_train):

        lv = losses_train['main'].eval(feed_dict=feed_train)

        if False:
            self.log_print('Batch %d, training loss %g' % (batch_idx, lv))
            for key, value in losses_train.items():
                self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_train)))

        i = batch_idx * self.batch

        self.attr_loss_table_train[batch_idx] = lv
        self.att_table_train[i:i+self.batch,:] = self.att_train.eval(feed_dict=feed_train)

    def update_val(self, batch_idx, feed_val, losses_train):

        lv = losses_train['main'].eval(feed_dict=feed_val)

        if False:
            self.log_print('Batch %d, Validation loss %g' % (batch_idx, lv))
            for key, value in losses_train.items():
                self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_val)))

        i = batch_idx * self.batch

        self.attr_loss_table_val[batch_idx] = lv
        self.att_table_val[i:i + self.batch, :] = self.att_train.eval(feed_dict=feed_val)

    def pre_epoch(self, graph):
        return

    def update_epoch(self, mesh_train):

        import numpy.linalg as LA

        #self.log_print('Training Loss: %g' % np.sqrt(LA.norm(self.att_table_train-y_truth_train)**2/(self.n_train*self.c_att)) )

        #self.log_print('')

        self.log_print('Average Training Loss: %g' % np.average(self.attr_loss_table_train))

    def update_epoch_val(self, mesh_val):

        import numpy.linalg as LA

        # self.log_print('Validation Loss: %g' % np.sqrt(LA.norm(self.att_table_val-y_truth_val)**2/(self.n_val*self.c_att)) )

        #self.log_print('')

        self.log_print('Average Validation Loss: %g' % np.average(self.attr_loss_table_val))


# This cannot be used directly, since it asks for n_class. An wrapper has to get around this.
class BaseClsTrainOp:

    def __init__(self, n_train, n_val, n_class, batch, out_train, log_print):

        self.n_train = n_train
        self.n_val = n_val
        self.n_class = n_class
        self.batch = batch
        self.out_train = out_train
        self.log_print = log_print

        self.logits_table_train = np.zeros([self.n_train, self.n_class])
        self.logits_table_val = np.zeros([self.n_val, self.n_class])

        #accuracy_train = ss.get_accuracy(mnist_train, logits_train)
        #accuracy_test  = ss.get_accuracy(mnist_test,  logits_test)

    def update(self, batch_idx, feed_train, losses_train):

        self.log_print('Batch %d, Training loss %g' % (batch_idx, losses_train['total'].eval(feed_dict=feed_train)))
        for key, value in losses_train.items():
            self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_train)))

        i = batch_idx * self.batch

        self.logits_table_train[i:i+self.batch,:] = self.out_train.logits.eval(feed_dict=feed_train)

    def pre_epoch(self, graph):
        return

    def update_val(self, batch_idx, feed_val, losses_train):

        self.log_print('Batch %d, Validation loss %g' % (batch_idx, losses_train['total'].eval(feed_dict=feed_val)))
        for key, value in losses_train.items():
            self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_val)))

        i = batch_idx * self.batch

        self.logits_table_val[i:i + self.batch, :] = self.out_train.logits.eval(feed_dict=feed_val)

    def update_epoch(self, mesh_train):

        # y_truth_train = mesh_train.label.np_data.TY
        # y_truth_train = y_truth_train[:, 0]

        y_truth_train = lref(mesh_train).get(range(mesh_train.get_num_data()))

        # print('update_epoch() get y_truth_train with size:', y_truth_train.shape)

        y_pred_train = np.argmax(self.logits_table_train, axis=1)

        y_pred_top_k_train = np.argsort(self.logits_table_train, axis=1)[:, ::-1]

        self.log_print('Training Accuracy: %g' % (np.sum(y_pred_train == y_truth_train) / self.n_train))

        self.log_print('Training Accuracy Top 5: %g' % (
        np.sum(y_pred_top_k_train[:, 0:5] == y_truth_train[:, np.newaxis]) / self.n_train))

        self.log_print('')

    def update_epoch_val(self, mesh_val):

        # TODO: change _test to _val

        # y_truth_test = mesh_val.label.np_data.TY
        # y_truth_test = y_truth_test[:, 0]

        y_truth_val = lref(mesh_val).get(range(mesh_val.get_num_data()))

        y_pred_val = np.argmax(self.logits_table_val, axis=1)

        y_pred_top_k_val = np.argsort(self.logits_table_val, axis=1)[:, ::-1]

        self.log_print('Validation Accuracy: %g' % (np.sum(y_pred_val == y_truth_val) / self.n_val))

        self.log_print(
            'Validation Accuracy Top 5: %g' % (np.sum(y_pred_top_k_val[:, 0:5] == y_truth_val[:, np.newaxis]) / self.n_val))


class BaseClsTrainOp2:

    def __init__(self, n_train, n_val, n_class, batch, out_train, log_print):

        self.n_train = n_train
        self.n_val = n_val
        self.n_class = n_class
        self.batch = batch
        self.out_train = out_train
        self.log_print = log_print

        self.logits_table_train = np.zeros([self.n_train, self.n_class])
        self.logits_table_val = np.zeros([self.n_val, self.n_class])

        # accuracy_train = ss.get_accuracy(mnist_train, logits_train)
        # accuracy_test  = ss.get_accuracy(mnist_test,  logits_test)

    def update(self, batch_idx, feed_train, losses_train):

        self.log_print('Batch %d, Training loss %g' % (batch_idx, losses_train['total'].eval(feed_dict=feed_train)))
        for key, value in losses_train.items():
            self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_train)))

        i = batch_idx * self.batch

        self.logits_table_train[i:i + self.batch, :] = self.out_train.logits.eval(feed_dict=feed_train)

    def pre_epoch(self, graph):
        return

    def update_val(self, batch_idx, feed_val, losses_train):

        self.log_print('Batch %d, Validation loss %g' % (batch_idx, losses_train['total'].eval(feed_dict=feed_val)))
        for key, value in losses_train.items():
            self.log_print(' %s=%g' % (key, value.eval(feed_dict=feed_val)))

        i = batch_idx * self.batch

        self.logits_table_val[i:i + self.batch, :] = self.out_train.logits.eval(feed_dict=feed_val)

    def update_epoch(self, mesh_train):

        # y_truth_train = mesh_train.label.np_data.TY
        # y_truth_train = y_truth_train[:, 0]

        # out = mesh_train.mesh.tf_data.get(range(mesh_train.get_num_data()))
        out = mesh_train.mesh.provider.get(range(mesh_train.get_num_data()))
        y_truth_train = out['label']
        y_truth_train = y_truth_train[:, 0]
        print('update_epoch(): y_truth_train: ', y_truth_train)

        # print('update_epoch() get y_truth_train with size:', y_truth_train.shape)

        y_pred_train = np.argmax(self.logits_table_train, axis=1)

        y_pred_top_k_train = np.argsort(self.logits_table_train, axis=1)[:, ::-1]

        self.log_print('Training Accuracy: %g' % (np.sum(y_pred_train == y_truth_train) / self.n_train))

        self.log_print('Training Accuracy Top 5: %g' % (
            np.sum(y_pred_top_k_train[:, 0:5] == y_truth_train[:, np.newaxis]) / self.n_train))

        self.log_print('')

    def update_epoch_val(self, mesh_val):

        # TODO: change _test to _val

        # y_truth_test = mesh_val.label.np_data.TY
        # y_truth_test = y_truth_test[:, 0]

        # out = mesh_val.get(range(mesh_val.get_num_data()))
        out = mesh_val.mesh.provider.get(range(mesh_val.get_num_data()))
        y_truth_val = out['label']
        y_truth_val = y_truth_val[:, 0]

        y_pred_val = np.argmax(self.logits_table_val, axis=1)

        y_pred_top_k_val = np.argsort(self.logits_table_val, axis=1)[:, ::-1]

        self.log_print('Validation Accuracy: %g' % (np.sum(y_pred_val == y_truth_val) / self.n_val))

        self.log_print(
            'Validation Accuracy Top 5: %g' % (
            np.sum(y_pred_top_k_val[:, 0:5] == y_truth_val[:, np.newaxis]) / self.n_val))


#class ClsTrainOp():
#    def __init__(self, n_class):


class SpecialClsTrainOp40(BaseClsTrainOp):
    def __init__(self, n_train, n_val, batch, logits_train, log_print):
        n_class = 40
        BaseClsTrainOp.__init__(self, n_train, n_val, n_class, batch, logits_train, log_print)


class SpecialClsTrainOp50(BaseClsTrainOp):
    def __init__(self, n_train, n_val, batch, logits_train, log_print):
        n_class = 50
        BaseClsTrainOp.__init__(self, n_train, n_val, n_class, batch, logits_train, log_print)


class SpecialClsTrainOp10(BaseClsTrainOp):
    def __init__(self, n_train, n_val, batch, logits_train, log_print):
        n_class = 10
        BaseClsTrainOp.__init__(self, n_train, n_val, n_class, batch, logits_train, log_print)


def struct2tf(mesh_all, level):
    struct_tf = su.C()
    struct_tf.mesh = mesh_all.mesh.tf_data
    if level >= 1:
        struct_tf.mesh_d = mesh_all.mesh_d.tf_data
    if level >= 2:
        struct_tf.mesh_d2 = mesh_all.mesh_d2.tf_data
    if level >= 3:
        struct_tf.mesh_d3 = mesh_all.mesh_d3.tf_data
    if level >= 4:
        struct_tf.mesh_d4 = mesh_all.mesh_d4.tf_data

    if level >= 1:
        struct_tf.ds01 = mesh_all.ds01.tf_data
    if level >= 2:
        struct_tf.ds12 = mesh_all.ds12.tf_data
    if level >= 3:
        struct_tf.ds23 = mesh_all.ds23.tf_data
    if level >= 4:
        struct_tf.ds34 = mesh_all.ds34.tf_data

    return struct_tf


def ModelWrapper(model, level, *positional_parameters, **keyword_parameters):
    #self.model = model
    #self.level = level

    # convert arguments to a dict.
    options = su.G()
    for kp in keyword_parameters:
        setattr(options, kp, keyword_parameters[kp])

    def model_wrapper(mesh_struct, metadata, cmd_option):
        assert (cmd_option == '')  # not implemented to add it from command line yet.

        return model(struct2tf(mesh_struct, level), metadata.tf, options)

    return model_wrapper

# class ModelWrapper():


BASE_LEARNING_RATE = 0.001
DECAY_STEP = 200000
DECAY_RATE = 0.7


def get_learning_rate(tf_batch_idx, batch_size, base_learning_rate):#, BASE_LEARNING_RATE, DECAY_STEP=200000, DECAY_RATE=0.7):
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        tf_batch_idx * batch_size,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99


# import tensorflow as tf
def get_bn_decay(tf_batch_idx, batch_size): #, BN_INIT_DECAY=0.5, BN_DECAY_DECAY_STEP=BN_DECAY_DECAY_STEP, BN_DECAY_DECAY_RATE=float(200000), BN_DECAY_CLIP=0.99):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        tf_batch_idx * batch_size,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


class MetaData():

    def __init__(self):
        return

    def init_tf(self, batch_size, base_lr=BASE_LEARNING_RATE):
        global_step = tf.Variable(0)
        bn_decay = get_bn_decay(global_step, batch_size)
        learning_rate = get_learning_rate(global_step, batch_size, base_lr)

        self.tf = su.G(
            global_step=global_step,
            is_training=tf.placeholder(tf.bool, shape=()),
            # derived tf data, does not need feed.
            bn_decay=bn_decay,
            learning_rate=learning_rate)

        # some metadata of tf data

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.

        tf.summary.scalar('bn_decay', bn_decay)
        tf.summary.scalar('learning_rate', learning_rate)

    def make_batch(self, batch_index, is_training):

        # global_step does not need to be feed since passed in the optimizer.
        return {self.tf.is_training: is_training}


#class GlobalInfo():
#    def __init__(self):
#        self.learning_rate = 0.01

class Optimizer():
    def __init__(self, losses, metadata):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=metadata.tf.learning_rate)
        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        self.train_step = self.optimizer.minimize(losses['total'], global_step=metadata.tf.global_step)



'''
def get(self, options=su.C()):

    def model_wrapper( mesh_struct, cmd_option):
        assert(self.level==2)
        assert(cmd_option=='') # not implemented to add it from command line yet.

        return self.model( struct2tf( mesh_struct, self.level), options)

    return model_wrapper
'''

