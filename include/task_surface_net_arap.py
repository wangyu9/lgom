import surf_struct as ss

MESH_LEVEL = 0

#MAX_NUM_VERTICES = 1024
#MAX_NUM_FACES = 2048
MAX_NUM_VERTICES = 640 # 1024
MAX_NUM_FACES = 1200 # 2048

def smooth_l1_loss(outputs, targets):
    # the smooth_l1_loss in torch in the huber_loss in tf.
    # https://pytorch.org/docs/0.4.1/nn.html?highlight=smooth_l1_loss#torch.nn.SmoothL1Loss
    # https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    import tensorflow as tf

    return tf.reduce_sum(tf.losses.huber_loss(labels=targets, predictions=outputs, reduction=tf.losses.Reduction.NONE)) # node the reverse of order.


def mesh_attr_loss(mesh_struct, out):
    mesh0 = mesh_struct.mesh.tf_data
    import surf_dg as sd
    import surf_basic as sb
    batch_size = sb.get_tensor_shape(mesh0.YVA)[0]

    # todo: consider change naming of logits.
    # return ss.attr_loss(None, attr_pred=out.logits, attr_truth=mesh0.YVA)

    weights_decay = 1e-5

    # https://stackoverflow.com/questions/38286717/tensorflow-regularization-with-l2-loss-how-to-apply-to-all-weights-not-just
    import tensorflow as tf
    vars = tf.trainable_variables()
    pattern_end = 'feature_linear_trans/weights:0'
    w_list = [var for var in vars if var.name[-len(pattern_end):] == pattern_end]
    flt_weights_l2 = tf.constant(0, dtype=tf.float32) if len(w_list)==0 else tf.add_n([tf.nn.l2_loss(w) for w in w_list])
    # this does not cover the ones in e.g. Linear, which has different names.

    return {'main': smooth_l1_loss(outputs=out.logits, targets=mesh0.YVA)/batch_size, 'weights_decay': (out.weights_l2_sum + flt_weights_l2) * weights_decay} #, 'total_weights'}


def get_loss():
    return mesh_attr_loss


def get_train_op():
    return ss.AttrTrainOp


def get_mesh_level():
    return MESH_LEVEL


def get_dataset():
    return 'data_plus'


def get_mesh_struct():
    return MeshStructSN


#########################################################

pass_sparse_ops = True
use_sparse_IJV = True

def SN_create_mesh_tf(batch, provider):
    import tensorflow as tf

    import surf_util as su
    mesh = su.C()

    n = provider.n
    f = provider.f
    ta = provider.ta
    tb = provider.tb
    yva = provider.yva

    mesh.V = tf.placeholder(tf.float32, shape=[batch, n, 3], name='V')
    mesh.F = tf.placeholder(tf.int32, shape=[batch,f,3], name='F')

    mesh.A = tf.placeholder(tf.float32, shape=[batch,n,ta], name='A')
    mesh.b = tf.placeholder(tf.float32, shape=[batch,tb], name='b')

    mesh.Mask = tf.placeholder(tf.float32, shape=[batch, n, 1], name='Mask')

    mesh.YVA = tf.placeholder(tf.float32, shape=[batch, n, yva], name='YVA')

    if pass_sparse_ops:
        if use_sparse_IJV:
            def create_sparse(mesh, name, shape):
                r = tf.placeholder(tf.int64, shape=[None], name=name + '_r')
                c = tf.placeholder(tf.int64, shape=[None], name=name + '_c')
                v = tf.placeholder(tf.float32, shape=[None], name=name + '_v')

                setattr(mesh, name + '_r', r)
                setattr(mesh, name + '_c', c)
                setattr(mesh, name + '_v', v)
                setattr(mesh, name, tf.SparseTensor(indices=tf.stack([r, c], axis=-1), values=v, dense_shape=shape))

            for i in range(batch):
                create_sparse(mesh, name='L_%d'%i, shape=[n, n])
                create_sparse(mesh, name='Di_%d'%i, shape=[4*f, 4*n])
                create_sparse(mesh, name='DiA_%d'%i, shape=[4*n, 4*f])
            mesh.L = [getattr(mesh, 'L_%d' % i) for i in range(batch)]
            mesh.Di = [getattr(mesh, 'Di_%d' % i) for i in range(batch)]
            mesh.DiA = [getattr(mesh, 'DiA_%d' % i) for i in range(batch)]
        else:
            # https://github.com/tensorflow/tensorflow/issues/6749
            import numpy as np
            mesh.L = [tf.sparse_placeholder(tf.float32, shape=np.array([n, n], dtype=np.int32), name='L_%d'%i) for i in range(batch)]
            mesh.Di = [tf.sparse_placeholder(tf.float32, shape=np.array([4*f, 4*n], dtype=np.int32), name='Di_%d' % i) for i in range(batch)]
            mesh.DiA = [tf.sparse_placeholder(tf.float32, shape=np.array([4*n, 4*f], dtype=np.int32), name='DiA_%d' % i) for i in range(batch)]

    return mesh


def to_tf_sparse(sp):
    import tensorflow as tf
    tf.SparseTensorValue(sp)

# TODO: automate this process:
def SN_make_mesh_feed_list(tf_data, provider, out):

    feed_list = {
        tf_data.V: out['V'],
        tf_data.F: out['F'],
        tf_data.A: out['A'],
        tf_data.b: out['b'],
        tf_data.Mask: out['Mask'],
        tf_data.YVA: out['YVA'],
    }

    batch_size = out['V'].shape[0]

    # print('SN_make_mesh_feed_list: batch_size=', batch_size) this is correct.
    if pass_sparse_ops:
        if use_sparse_IJV:
            def update_feed_list(feed_list, tf_data, name, out):
                # feed_list.update({getattr(tf_data, name+'_r'):out[name+'_r']}) # tf_data.L[i]: out['L_%d' % i]})
                feed_list.update({getattr(tf_data, name + '_r'): out[name + '_r']})
                feed_list.update({getattr(tf_data, name + '_c'): out[name + '_c']})
                feed_list.update({getattr(tf_data, name + '_v'): out[name + '_v']})
            for i in range(batch_size):
                update_feed_list(feed_list, tf_data, 'L_%d' % i, out)
                update_feed_list(feed_list, tf_data, 'Di_%d' % i, out)
                update_feed_list(feed_list, tf_data, 'DiA_%d' % i, out)
        else:
            for i in range(batch_size):#
            #for i in range(1000000): # max_possible batch_size
                if True: #
                # if ('L-%d'%i) in out:
                #This is wrong!!! if hasattr(out, 'L_%d'%i):
                    feed_list.update({tf_data.L[i]: out['L_%d' % i]})
                    feed_list.update({tf_data.Di[i]: out['Di_%d' % i]})
                    feed_list.update({tf_data.DiA[i]: out['DiA_%d' % i]})

    return feed_list


# TODO: remove all meshbatch classes, since it is not used anyway
class SN_MeshBatch:
    def __init__(self, provider, out):

        self.TV = None
        self.TF = None
        self.TA = None
        self.Tb = None

        return


load_all_at_once = False


class SN_MeshProvider:
    # n, f, ta, tb
    def print(self, s):
        print('MeshProvider(): %s'%(s))

    def __init__(self, filename, phase, level, max_adj):
        self.filename = filename
        self.phase = phase
        self.level = level

        # import task_surface_net_arap
        self.n = MAX_NUM_VERTICES # task_surface_net_arap
        self.f = MAX_NUM_FACES # task_surface_net_arap
        self.ta = 6
        self.tb = 1
        self.yva = 120

        if load_all_at_once:
            self.sequences = read_data_fast(filename, 'dir', phase) # task_surface_net_arap
            analyze_seq(self.sequences) # task_surface_net_arap
        else:
            self.meta, self.sequences = read_data_meta(filename, 'dir', phase)

        return

    def get(self, r, augmentor=None):

        #if augmentor==None:
        #    augmentor = Augmentor()

        # import task_surface_net_arap
        random_offsets = (self.phase=='train')
        if load_all_at_once:
            V, A, Targets, Mask, F, L, Di, DiA = get_batch( # task_surface_net_arap
                                        self.sequences, r,
                                        num_vertices=MAX_NUM_VERTICES, # task_surface_net_arap
                                        num_faces=MAX_NUM_FACES, # task_surface_net_arap
                                        random_offsets=random_offsets)
        else:
            # this modified sequences which is a list,
            # todo: have a better API for it.
            V, A, Targets, Mask, F, L, Di, DiA = get_batch2(  # task_surface_net_arap
                self.meta,
                self.sequences, r,
                num_vertices=MAX_NUM_VERTICES,  # task_surface_net_arap
                num_faces=MAX_NUM_FACES,
                random_offsets=random_offsets)

        batch_size = len(r)

        import numpy as np

        out = {'V': V,
                'F': F,
                'A': A,
                'b': np.zeros(shape=[batch_size,1]),
                'Mask': Mask,
                'YVA': Targets}

        if pass_sparse_ops:
            def update_out_sparse(out, name, tensor):
                import scipy
                import scipy.sparse
                X = scipy.sparse.coo_matrix(tensor)

                out.update({(name + '_r'): X.row})
                out.update({(name + '_c'): X.col})
                out.update({(name + '_v'): X.data})
                out.update({(name + '_shape'): X.shape})

            for i in range(batch_size):
                if use_sparse_IJV:
                    update_out_sparse(out, 'L_%d' % i, L[i])
                    update_out_sparse(out, 'Di_%d' % i, Di[i])
                    update_out_sparse(out, 'DiA_%d' % i, DiA[i])

                else:
                    out.update({('L_%d'%i):L[i]})
                    out.update({('Di_%d'%i):Di[i]})
                    out.update({('DiA_%d'%i):DiA[i]})

        return out

    def getV(self, r):
        assert False


def SN_MeshStructMetaReader(filename, mesh_level, phase):

    level = 0
    # import task_surface_net_arap
    if load_all_at_once:
        sequences = read_data_fast(filename, 'dir', phase) # task_surface_net_arap
    else:
        meta, sequences = read_data_meta(filename, 'dir', phase)

    # sequences have None of the right length.
    num_data = len(sequences)

    return level, num_data

'''
class SN_DownProvider:

    def print(self, s):
        print('DownProvider(): %s'%(s))

    def __init__(self, h5, phase, level, mesh_provider=None, mesh_provider_d=None):

        return

    def get(self, r):

        return self.rDSI.get(r), self.rDSW.get(r)


class SN_LabelProvider:

    def print(self, s):
        print('LabelProvider(): %s'%(s))

    def __init__(self, h5, phase):

        return

    def get(self, r):

        return self.rY.get(r)
'''



Mesh = ss.MeshGeneric(SN_MeshBatch, SN_create_mesh_tf, SN_make_mesh_feed_list)
MeshProvider = SN_MeshProvider
MeshStructSN = ss.MeshStructGeneric(Mesh, SN_MeshStructMetaReader, MeshProvider, None, None)


#########################################################

#global sequences_loaded = None

def read_data_fast(mypyth, model, phase):
    if read_data_fast.sequences_loaded is None:
        read_data_fast.sequences_loaded = read_data(mypyth, model)
    if phase=='train':
        all = read_data_fast.sequences_loaded
        # return sequences[0:300] #300 #
        return all[0:(len(sequences)//10*8)]
    elif phase=='test':
        all = read_data_fast.sequences_loaded
        # return sequences[300:] # 10 #
        return all[(len(sequences)//10*8):]
    else:
        assert phase=='all'
        return read_data_fast.sequences_loaded


read_data_fast.sequences_loaded = None


def load_file_32(seqname, mypath, model):
    import numpy as np
    def d_32(A):
        return np.array(A, dtype=np.float32)

    import scipy
    import scipy.sparse
    def sp_32(A):
        return scipy.sparse.csr_matrix(A, shape=A.shape, dtype=np.float32)


    sequence = np.load(open(mypath + "/" + seqname, 'rb'), encoding='latin1')
    new_sequence = []
    for i, frame in enumerate(sequence):

        frame['V'] = d_32(frame['V'])
        frame['F'] = d_32(frame['F'])

        # frame['V'] = torch.from_numpy(frame['V'])
        # frame['F'] = torch.from_numpy(frame['F'])
        # frame['V'], frame['F'] # in numpy format.
        if i < 10:
            frame['L'] = sp_32(frame['L'])  # utils.sp_sparse_to_pt_sparse(frame['L'])

            # print(frame['L'])

            if model == "dir":
                frame['Di'] = sp_32(frame['Di'])  # utils.sp_sparse_to_pt_sparse(frame['Di'])
                frame['DiA'] = sp_32(frame['DiA'])  # utils.sp_sparse_to_pt_sparse(frame['DiA'])
            else:
                frame['Di'] = None
                frame['DiA'] = None
        new_sequence.append(frame)

    return new_sequence


def load_file_0(seqname, mypath, model):
    import numpy as np
    sequence = np.load(open(mypath + "/" + seqname, 'rb'), encoding='latin1')
    new_sequence = []
    for i, frame in enumerate(sequence):

        # frame['V'] = torch.from_numpy(frame['V'])
        # frame['F'] = torch.from_numpy(frame['F'])
        # frame['V'], frame['F'] # in numpy format.
        if i < 10:
            frame['L'] = frame['L']  # utils.sp_sparse_to_pt_sparse(frame['L'])

            # print(frame['L'])

            if model == "dir":
                frame['Di'] = frame['Di']  # utils.sp_sparse_to_pt_sparse(frame['Di'])
                frame['DiA'] = frame['DiA']  # utils.sp_sparse_to_pt_sparse(frame['DiA'])
            else:
                frame['Di'] = None
                frame['DiA'] = None
        new_sequence.append(frame)

    return new_sequence


def read_data(mypath, model, load_file=load_file_32):
    # output:
    # The sequences data structure:
    # [num_data][num_frames=50]['V']

    from os import listdir
    from os.path import isdir, isfile, join
    import progressbar as pb
    import numpy as np
    import gc

    files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith("npy"))])

    print("Loading the dataset")
    pbar = pb.ProgressBar()

    sequences = []

    for seqname in pbar(files):
        sequences.append(load_file(seqname, mypath, model))

        if len(sequences) % 100 == 0:
            gc.collect()

    return sequences


def read_batch(sequences, indices, offsets, batch_size, num_vertices, num_faces, input_frames, output_frames, read_dir):
    # Output:
    #
    # (5, 1024, 6) inputs,
    # (5, 1024, 120) targets,
    # (5, 1024, 1) mask,
    # (5, 2048, 3) faces,
    # (464, 464) * 5 as list laplacian,
    # (3476, 1856) * 5 as list Di
    # (1856, 3476) * 5 as list DiA


    import numpy as np
    inputs = np.zeros([batch_size, num_vertices, 3 * input_frames])
    targets = np.zeros([batch_size, num_vertices, 3 * output_frames])
    mask = np.zeros([batch_size, num_vertices, 1])
    faces = np.zeros([batch_size, num_faces, 3])

    laplacian = []

    Di = []
    DiA = []

    for b, (ind, offset) in enumerate(zip(indices, offsets)):
        # in original code: #offset = 0
        num_vertices = sequences[ind][0]['V'].shape[0] #size(0)
        num_faces = sequences[ind][0]['F'].shape[0] # size(0)

        for i in range(input_frames):
            inputs[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset]['V']

        for i in range(output_frames):
            targets[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset + input_frames]['V']

        mask[b, :num_vertices] = 1
        faces[b, :num_faces] = sequences[ind][0]['F']

        L = sequences[ind][offset + input_frames - 1]['L']
        laplacian.append(L)

        if read_dir:
            Di.append(sequences[ind][offset + input_frames - 1]['Di'])
            DiA.append(sequences[ind][offset + input_frames - 1]['DiA'])

    return inputs, targets, mask, faces, laplacian, Di, DiA


def read_batch_raw(sequences, indices, offsets, batch_size, num_vertices, num_faces, input_frames, output_frames, read_dir):
    # Output:
    #
    # (5, 1024, 6) inputs,
    # (5, 1024, 120) targets,
    # (5, 1024, 1) mask,
    # (5, 2048, 3) faces,
    # (464, 464) * 5 as list laplacian,
    # (3476, 1856) * 5 as list Di
    # (1856, 3476) * 5 as list DiA


    import numpy as np
    inputs = [] # [[] for i in range(batch_size)] # np.zeros([batch_size, num_vertices, 3 * input_frames])
    targets = [] # [[] for i in range(batch_size)] # np.zeros([batch_size, num_vertices, 3 * output_frames])
    mask = [] # [[] for i in range(batch_size)]  # np.zeros([batch_size, num_vertices, 1])
    faces = [] # [[] for i in range(batch_size)] # np.zeros([batch_size, num_faces, 3])

    laplacian = []

    Di = []
    DiA = []

    for b, (ind, offset) in enumerate(zip(indices, offsets)):

        if False:
            # in original code: #offset = 0
            num_vertices = sequences[ind][0]['V'].shape[0] #size(0)
            num_faces = sequences[ind][0]['F'].shape[0] # size(0)

            for i in range(input_frames):
                inputs[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset]['V']

            for i in range(output_frames):
                targets[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset + input_frames]['V']

            mask[b, :num_vertices] = 1
            faces[b, :num_faces] = sequences[ind][0]['F']

            L = sequences[ind][offset + input_frames - 1]['L']
            laplacian.append(L)

            if read_dir:
                Di.append(sequences[ind][offset + input_frames - 1]['Di'])
                DiA.append(sequences[ind][offset + input_frames - 1]['DiA'])

        else:

            num_vertices = sequences[ind][0]['V'].shape[0] #size(0)
            num_faces = sequences[ind][0]['F'].shape[0] # size(0)

            V = np.zeros([num_vertices, 3 * input_frames])
            T = np.zeros([num_vertices, 3 * output_frames])
            M = np.zeros([num_vertices, 1])
            F = np.zeros([num_faces, 3])

            for i in range(input_frames):
                V[:num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset]['V']

            for i in range(output_frames):
                T[:num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset + input_frames]['V']

            M[:num_vertices] = 1
            F[:num_faces] = sequences[ind][0]['F']

            inputs.append(V)
            targets.append(T)
            mask.append(M)
            faces.append(F)

            L = sequences[ind][offset + input_frames - 1]['L']
            laplacian.append(L)

            if read_dir:
                Di.append(sequences[ind][offset + input_frames - 1]['Di'])
                DiA.append(sequences[ind][offset + input_frames - 1]['DiA'])



    return inputs, targets, mask, faces, laplacian, Di, DiA


def get_batch(sequences, r, num_vertices, num_faces, random_offsets, raw=False):
    input_frames = 2
    output_frames = 40
    indices = [r[i] for i in range(len(r))]
    #indices = [r[i] for i in range(r.shape[0])]
    if random_offsets:
        import numpy
        import numpy.random
        offsets = [numpy.random.randint(0, len(sequences[ind]) - input_frames - output_frames) for ind in indices]
    else:
        # offsets = [0 for i in range(len(indices))]
        offsets = [(ind % (len(sequences[ind]) - input_frames - output_frames)) for ind in indices]


    batch_size = len(indices)

    if raw:
        data_batch = read_batch_raw(sequences, indices, offsets, batch_size, num_vertices, num_faces, input_frames=input_frames, output_frames=output_frames, read_dir=True)
        return data_batch

    data_batch = read_batch(sequences, indices, offsets, batch_size, num_vertices, num_faces, input_frames=input_frames, output_frames=output_frames, read_dir=True)



    V = data_batch[0][:, :, 3:6]
    A = data_batch[0] # (5, 1024, 6) inputs
    Targets = data_batch[1] # (5, 1024, 120) targets
    Mask = data_batch[2] # (5, 1024, 1) mask
    F = data_batch[3] # (5, 2048, 3) faces
    L = data_batch[4] # (464, 464) * 5 as list # cannot be used before reshaped to MAX_NUM_VERTICES x MAX_NUM_VERTICES
    Di = data_batch[5] # (3476, 1856) * 5  as list
    DiA = data_batch[6] # (1856, 3476) * 5 as list

    batch_size = A.shape[0]

    # (5, 1024, 6) inputs,
    # (5, 1024, 120) targets,
    # (5, 1024, 1) mask,
    # (5, 2048, 3) faces,
    # (464, 464) * 5 as list laplacian,
    # (3476, 1856) * 5 as list Di
    # (1856, 3476) * 5 as list DiA




    def pad_axis2(X, size_axis2):
        s = X.shape
        assert len(s)==3
        import numpy as np
        R = np.zeros(shape=[s[0],size_axis2,s[2]])
        R[:,0:s[1],:] = X
        return R


    V = pad_axis2(V, size_axis2=MAX_NUM_VERTICES) # batch_size, MAX_NUM_VERTICES, 3]
    A = pad_axis2(A, size_axis2=MAX_NUM_VERTICES)  # batch_size, MAX_NUM_VERTICES, 6]
    Targets = pad_axis2(Targets, size_axis2=MAX_NUM_VERTICES)
    Mask = pad_axis2(Mask, size_axis2=MAX_NUM_VERTICES)
    F = pad_axis2(F, size_axis2=MAX_NUM_FACES)

    def sparse_resize(X, new_shape):
        import scipy
        import scipy.sparse
        import numpy as np
        X = scipy.sparse.coo_matrix(X)
        shape = X.shape
        assert shape[0] <= new_shape[0]
        assert shape[1] <= new_shape[1]
        # this does not work: return scipy.sparse.csr_matrix((X.data, X.indices, X.indptr), shape=new_shape)
        return scipy.sparse.csr_matrix((X.data, (X.row, X.col)), shape=new_shape, dtype=np.float32)


    L = [sparse_resize(L[i], new_shape=[MAX_NUM_VERTICES, MAX_NUM_VERTICES]) for i in range(len(L))]
    Di = [sparse_resize(Di[i], new_shape=[4 * MAX_NUM_FACES, 4 * MAX_NUM_VERTICES]) for i in range(len(Di))]
    DiA = [sparse_resize(DiA[i], new_shape=[4 * MAX_NUM_VERTICES, 4 * MAX_NUM_FACES]) for i in range(len(DiA))]

    return V, A, Targets, Mask, F, L, Di, DiA


def read_data_meta(mypath, model, phase):
    # output:
    # The sequences data structure:
    # [num_data][num_frames=50]['V']

    from os import listdir
    from os.path import isdir, isfile, join
    import progressbar as pb
    import numpy as np
    import gc

    files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith("npy"))])

    print("Loading the dataset")
    pbar = pb.ProgressBar()

    sequences = []

    for seqname in pbar(files):
        sequences.append(None)

        if len(sequences) % 100 == 0:
            gc.collect()

    if phase=='train':
        # sequences = sequences[0:300] #
        sequences = sequences[0:(len(sequences)//10*8)]
    elif phase == 'test':
        # sequences = sequences[300:] # 10 #
        sequences = sequences[(len(sequences)//10*8):]
    else:
        assert phase == 'all'
        # sequences = sequences

    return [files, mypath, model], sequences


# this modifies sequences, consdier change the API
def read_data_range(mypath, model, files, sequences, r, load_file=load_file_32):
    # output:
    # The sequences data structure:
    # [num_data][num_frames=50]['V']

    from os import listdir
    from os.path import isdir, isfile, join
    import progressbar as pb
    import numpy as np
    import gc


    for i in r:# len(files):
        sequences[i] = load_file(files[i], mypath, model)

    return


def get_batch2(meta, sequences, r, num_vertices, num_faces, random_offsets, raw=False):
    [files, mypath, model] = meta
    # print("get_batch2", mypath, ' ', files, ' ', model)
    r2 = []
    for i in r:
        if sequences[i] is None:
            r2.append(i)
    read_data_range(mypath, model=model, files=files, sequences=sequences, r=r2)
    batch_data = get_batch(sequences, r, num_vertices, num_faces, random_offsets=random_offsets, raw=raw)
    for i in r:
        sequences[i] = None
    # erase the data so the memory does not explode
    return batch_data


def assume_less_or_eq(num, threshold):
    if num <= threshold:
        return
    else:
        print('%g is not less or eq to %g'%(num, threshold))
        assert False


def analyze_seq(sequences):

    num_meshes = len(sequences)

    num_frames = len(sequences[0])

    '''
    for i in range(num_data):
        print('mesh %d' % (i))
        num_frame_i = len(sequences[i])
        for j in range(num_frame_i):
            print('num_frames[%d]=%d' % (i, num_frame_i))
            print()
    '''

    import progressbar as pb
    import numpy as np
    import gc

    print("Analyzing the dataset")
    pbar = pb.ProgressBar()

    def analyze(animation):

        for i, frame in enumerate(animation):
            print('V:', frame['V'].shape)
            print('F:', frame['F'].shape)

            #frame['V'] = torch.from_numpy(frame['V'])
            #frame['F'] = torch.from_numpy(frame['F'])
            # frame['V'], frame['F'] # in numpy format.
            if i < 10:
                print('L:', frame['L'].shape)
                print('Di:', frame['Di'].shape)
                print('DiA:', frame['DiA'].shape)
                # frame['Di']
                # frame['DiA']

        return

    for i in range(len(sequences)):
        animation = sequences[i]
        frame = animation[0]
        assume_less_or_eq(frame['V'].shape[0], MAX_NUM_VERTICES)
        assume_less_or_eq(frame['F'].shape[0], MAX_NUM_FACES)

    for i in range(len(sequences)):
        animation = sequences[i]
        # analyze(animation)


FILE_PATH = 'D:/WorkSpace/vmware/shared/ubuntu/WorkSpace/dg/util/include/sn_arap_data_sample'

import sys


# example: python3 task_surface_net_arap.py ../../surfacenetworks/src/as_rigid_as_possible/data_plus
# argument:  ../../surfacenetworks/src/as_rigid_as_possible/data_plus

if  __name__ == "__main__" or '__file__' not in globals():

    if False:

        # this used too much memory ~60GB+.

        sequences = read_data_fast(FILE_PATH, model='dir', phase='train')


        import graphics as gx
        import numpy as np

        idm = 0
        idf = 49
        V = sequences[idm][idf]['V']
        F = sequences[idm][idf]['F']
        #L = sequences[idm][idf]['L']
        #Di = sequences[idm][idf]['Di']
        #DiA = sequences[idm][idf]['DiA']


        gx.show_mesh(V,F)

        # r = get_batch(sequences, np.arange(5), MAX_NUM_VERTICES, MAX_NUM_FACES)


        seq_train = read_data_fast(FILE_PATH, model='dir', phase='train')

        V, A, Targets, Mask, F, L, Di, DiA = get_batch(seq_train, [i for i in range(32)],
                      num_vertices=MAX_NUM_VERTICES,
                      num_faces=MAX_NUM_FACES,  random_offsets=True)


        import numpy as np
        # V = sequences[0][0]['V']
        # V = r[1]
        # V = np.random.rand(32,5,3)


        import tensorflow as tf
        V_tf = tf.convert_to_tensor(V)
        with tf.Session():
            tfv = smooth_l1_loss(tf.zeros_like(V_tf), V_tf).eval()

        import torch
        import torch.nn
        from torch.nn.functional import smooth_l1_loss as smooth_l1_loss_pt

        from torch.autograd import Variable
        V_pt = Variable(torch.from_numpy(V))
        ptv = smooth_l1_loss_pt(torch.zeros_like(V_pt), V_pt, size_average=False).data.cpu().numpy()

        print(ptv/tfv)

    else:
        # mypath = '../data/data_plus'
        # mypath = FILE_PATH
        mypath = sys.argv[1]
        meta, sequences = read_data_meta(mypath, 'dir', 'all')

        batch_size = 50

        r = [i for i in range(5)]
        for i in range(len(sequences)//batch_size):
            r = range(i*batch_size, (i+1)*batch_size)

            read_data_range(meta[1], model=meta[2], files=meta[0], sequences=sequences, r=r)
            # V, A, Targets, Mask, F, L, Di, DiA = get_batch(sequences, r, num_vertices=MAX_NUM_VERTICES, num_faces=MAX_NUM_FACES, random_offsets=False, raw=True)

            inputs, targets, mask, faces, laplacian, Di, DiA = get_batch2(
                meta,
                sequences, r,
                num_vertices=MAX_NUM_VERTICES,
                num_faces=MAX_NUM_FACES,
                random_offsets=False,
                raw=True)

            n_min = min([inputs[i].shape[0] for i in range(len(r))])
            n_max = max([inputs[i].shape[0] for i in range(len(r))])

            f_min = min([faces[i].shape[0] for i in range(len(r))])
            f_max = max([faces[i].shape[0] for i in range(len(r))])

            assert n_max < MAX_NUM_VERTICES
            assert f_max < MAX_NUM_FACES

            V, A, Targets, Mask, F, L, Di, DiA = get_batch2(
                meta,
                sequences, r,
                num_vertices=MAX_NUM_VERTICES,
                num_faces=MAX_NUM_FACES,
                random_offsets=False,
                raw=False)

            ## import tensorflow as tf
            import numpy as np
            minV = np.min(V, axis=(0,1))
            maxV = np.max(V, axis=(0,1))
            print('n: [%4d,%4d]'%(n_min,n_max), 'f: [%4d,%4d]'%(f_min,f_max), 'Data Range', minV, maxV)









def sample_batch(x):
    # todo
    sample_batch.num_vertices = 5
    print(sample_batch.num_vertices)
    return

