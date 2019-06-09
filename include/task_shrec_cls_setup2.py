###################################################
# Basic Definition Level
###################################################
import surf_struct as ss

NUM_CLASS = 50
MESH_LEVEL = 0

# TO implemented in its realization.
# def get_model():
#


# not implemented
# def get_augmentor():



def mesh_cls_loss_wd(mesh_struct, out):
    n_class = NUM_CLASS

    weights_decay = 1e-5
    import tensorflow as tf
    vars = tf.trainable_variables()
    pattern_end = 'feature_linear_trans/weights:0'
    w_list = [var for var in vars if var.name[-len(pattern_end):] == pattern_end]
    flt_weights_l2 = tf.add_n([tf.nn.l2_loss(w) for w in w_list])

    return {'main': cls_loss_local(mesh_struct, out.logits, n_class), 'weights_decay': flt_weights_l2 * weights_decay}

    # old
    # return ss.cls_loss(mesh_struct, out.logits, n_class)


def mesh_cls_loss(mesh_struct, out):
    n_class = NUM_CLASS

    return cls_loss_local(mesh_struct, out.logits, n_class)


def get_loss():
    return mesh_cls_loss


def get_train_op():
    #return ss.SpecialClsTrainOp50
    return SpecialClsTrainOp_local


def get_mesh_level():
    return MESH_LEVEL


def get_dataset():
    return 'shrec15-8k'
###################################################
# Wrapper Level
###################################################

# from model_cls_lap import *


# sio.loadmat('D:\\WorkSpace\\vmware\\shared\\ubuntu\\WorkSpace\\dg\\util\\include\\shrec15\\y_label.mat')['y_label'][:,0]


def get_mesh_struct():
    return MeshStructSN


class SpecialClsTrainOp_local(ss.BaseClsTrainOp2):
    def __init__(self, n_train, n_val, batch, logits_train, log_print):
        n_class = NUM_CLASS
        ss.BaseClsTrainOp2.__init__(self, n_train, n_val, n_class, batch, logits_train, log_print)


def cls_loss_local(mesh_struct, logits, n_class):
    label = mesh_struct.mesh.tf_data.label[:, 0]
    return ss.cls_loss_base(label, logits, n_class)


#####################



MAX_NUM_VERTICES = 4008 # 800
MAX_NUM_FACES = 8016 # 1600
#NUM_TRAIN = 60000
#NUM_TEST = 10000


def SN_create_mesh_tf(batch, provider):
    import tensorflow as tf

    import surf_util as su
    mesh = su.C()

    n = provider.n
    f = provider.f
    ta = provider.ta
    tb = provider.tb

    mesh.V = tf.placeholder(tf.float32, shape=[batch, n, 3], name='V')
    mesh.F = tf.placeholder(tf.int32, shape=[batch,f,3], name='F')

    mesh.A = tf.placeholder(tf.float32, shape=[batch,n,ta], name='A')
    mesh.b = tf.placeholder(tf.float32, shape=[batch,tb], name='b')

    mesh.label = tf.placeholder(tf.int32, [batch, 1], name='label')

    return mesh


# TODO: automate this process:
def SN_make_mesh_feed_list(tf_data, provider, out):

    feed_list = {
        tf_data.V: out['V'],
        tf_data.F: out['F'],
        tf_data.A: out['A'],
        tf_data.b: out['b'],
        tf_data.label: out['label']
    }

    return feed_list


# TODO: remove all meshbatch classes, since it is not used anyway
class SN_MeshBatch:
    def __init__(self, provider, out):

        self.TV = None
        self.TF = None
        self.TA = None
        self.Tb = None

        return


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
        self.ta = 1
        self.tb = 1

        import scipy.io as sio
        import numpy as np

        all = sio.loadmat(filename+'/y_label.mat')['y_label'][:, 0]
        assert len(all) == 1200
        if phase=='train':
            self.label = all[0:960]
            self.num_data = len(self.label)
            self.meshes_id = np.arange(0,960)
        else:
            assert phase=='test'
            self.label = all[960:]
            self.num_data = len(self.label)
            self.meshes_id = np.arange(960, 1200)

        # array of size (num,)
        #
        # print('dataset',filename,' phase, num_data:',self.num_data)

        import numpy as np
        self.load = [False for i in range(self.num_data)]
        self.V = np.zeros([self.num_data, MAX_NUM_VERTICES, 3])
        self.F = np.zeros([self.num_data, MAX_NUM_FACES, 3], dtype=np.int64)
        self.A = np.zeros([self.num_data, MAX_NUM_VERTICES, 1])
        self.b = np.zeros(shape=[self.num_data,1])

        return

    def get(self, r, augmentor=None):

        if augmentor==None:
            import surf_struct as ss
            augmentor = ss.Augmentor()

        # todo

        batch_size = len(r)

        import mesh
        import os.path
        # print('get():',r, self.load)
        for i in range(len(r)):
            if not self.load[r[i]]:
                obj_filename = self.filename+'/ori/'+'/T%d.off'%(self.meshes_id[r[i]])
                vertices, faces = mesh.load_off(obj_filename)
                assert vertices.shape[0] > 100
                assert faces.shape[0] > 100
                self.V[r[i], :vertices.shape[0], :] = vertices
                self.F[r[i], :faces.shape[0], :] = faces
                self.load[r[i]] = True

        import numpy as np
        out = {'V': augmentor.apply(self.V[r,:,:]),
                'F': self.F[r,:,:],
                'A': self.A[r,:,:],
                'b': self.b[r,:],
                'label': (self.label[r])[:,np.newaxis]} # [batch,] -> [batch,1]

        return out

    def getV(self, r):
        assert False


def SN_MeshStructMetaReader(filename, mesh_level, phase):

    level = 0
    import scipy.io as sio
    all = sio.loadmat(filename + '/y_label.mat')['y_label'][:, 0]

    assert len(all) == 1200
    if phase == 'train':
        num_data = 960
    else:
        assert phase == 'test'
        num_data = 240


    # if phase=="train":
    #     num_data = NUM_TRAIN
    # else:
    #     assert phase=="test"
    #     num_data = NUM_TEST

    return level, num_data




Mesh = ss.MeshGeneric(SN_MeshBatch, SN_create_mesh_tf, SN_make_mesh_feed_list)
MeshProvider = SN_MeshProvider
MeshStructSN = ss.MeshStructGeneric(Mesh, SN_MeshStructMetaReader, MeshProvider, None, None)



