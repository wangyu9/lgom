import numpy as np
import scipy

import util as util


class C(object):
    def __init__(self, keys='', values=[]):
        for (key, value) in zip(keys, values):
            self.__dict__[key] = value


class G():
    def __init__(self, *positional_parameters, **keyword_parameters):
        for kp in keyword_parameters:
            setattr(self, kp, keyword_parameters[kp])
        return


NOT_GIVEN = 'NotGiven'

def assume_attr(object, field, default=NOT_GIVEN):
    if default is NOT_GIVEN:
        assert (hasattr(object, field))
    else:
        if not hasattr(object, field):
            setattr(object, field, default)


from graphics import show_mesh

def show_mesh_tensor(TV,TF,Tu=None,fig=None,ax=None, TVC=None): # TVC):#

    rf = np.argwhere(TF[:,0]>=-0.0001)
    
    # print(rf.shape) #[F.shape[0],1]
    
    F = TF[rf[:,0],:].astype(int)
    
    print(F.shape)

    n = np.max(F) + 1
    V = TV[0:n,:]

    if TVC is None:
        if Tu is None:
            Tu = TV[:,0]
        return show_mesh(V,F,u=Tu,fig=fig,ax=ax)
    else:
        return show_mesh(TV,F,vertex_color=TVC,fig=fig,ax=ax)


def adjacent_faces_per_vertex(n,F):
    if F.shape[1]!= 3:
        print('F shape:',F.shape, F)
    assert ( F.shape[1]==3)
    J = [[] for i in range(n)]
    K = [[] for i in range(n)]
    for i in range(F.shape[0]):
        J[F[i,0]].append(i)
        J[F[i,1]].append(i)
        J[F[i,2]].append(i)
        K[F[i,0]].append(0)
        K[F[i,1]].append(1)
        K[F[i,2]].append(2)
    # this was the old one is there is probably a critical bug due to missing []: max_ajs = max([len(J[i])] for i in range(len(J)))[0]
    ajs = [len(J[i]) for i in range(len(J))]
    max_ajs = max(ajs)
    return J, K, max_ajs


def meshes_adjacenies(Vs,Fs,a=200,names=None):
    m = len(Vs)
    max_num_ajs = 0
    Js = []
    Ks = []
    for i in range(m): 
        V, F = Vs[i], Fs[i]        
        J, K, num_ajs = adjacent_faces_per_vertex(V.shape[0],F)
        if num_ajs > a - 1:
            if names==None:
                name = 'Unknown'
            else:
                name = names[i]
            print('Warning: Mesh[%d] with name %s has adjacency=%d'%(i,name,num_ajs))
        max_num_ajs = max(num_ajs, max_num_ajs)
        Js.append(J)
        Ks.append(K)
    print(max_num_ajs)
    return Js, Ks


def meshes_adjacenies_compact(n, Fs):
    m = len(Fs)
    Js = []
    Ks = []
    for i in range(m):
        F = Fs[i]
        J, K, num_ajs = adjacent_faces_per_vertex(n, F)
        Js.append(J)
        Ks.append(K)
    return Js, Ks


def meshes_to_tensor(Vs,Fs,Js,Ks,n,f,a, minus_unused_indices=True):
    
    m = len(Vs)
    assert(m==len(Fs))
    assert(m==len(Js))
    
    TV = np.zeros([m,n,3])

    if minus_unused_indices:
        TF = - np.ones([m,f,3])
        TJ = - np.ones([m,n,a])
        TK = - np.ones([m,n,a])
    else:
        TF = (n-1) * np.ones([m,f,3])
        TJ = (f-1) * np.ones([m,n,a])
        TK = (3-1) * np.ones([m,n,a])
    
    for i in range(m): 
        
        assert(n>Vs[i].shape[0])
        assert(f>Fs[i].shape[0])
        Ji = Js[i]
        Ki = Ks[i]
        max_ajs = max([len(Ji[j])] for j in range(len(Ji)))[0]
        if(a<max_ajs):
            print('Error: max_ajs: %d is larger than a(%d)'%(max_ajs,a))
        assert(a>=max_ajs)
        
        Vi = Vs[i]
        Fi = Fs[i]
        ni = Vi.shape[0]
        TV[i,0:ni,:] = Vi

        fi = Fi.shape[0]
        TF[i,0:fi,:] = Fi
        
        assert(Vi.shape[0]==len(Ji))
        
        for j in range(len(Ji)):
            for k in range(len(Ji[j])):
                TJ[i,j,k] = Ji[j][k]
                TK[i,j,k] = Ki[j][k]
    return TV, TF, TJ, TK


def meshes_attributes_to_tensor(As,n):
    
    m = len(As)
    c = As[0].shape[1]
    
    TA = np.zeros([m,n,c])
    
    for i in range(m):
        
        util.assert_equal(c, As[i].shape[1])
        
        ni = As[i].shape[0]
        TA[i,0:ni,:] = As[i]
        
    return TA


def meshes_info_to_tensor(bs):
    m = len(bs)
    c = bs[0].shape[0]

    Tb = np.zeros([m, c])

    for i in range(m):
        util.assert_equal(c, bs[i].shape[0])

        Tb[i,:] = bs[i]

    return Tb


def meshes_index_attributes_to_tensor(As, n):
    m = len(As)
    c = As[0].shape[1]

    TA =  np.zeros([m, n, c], np.int32) # as -1, here is different from meshes_attributes_to_tensor

    for i in range(m):
        util.assert_equal(c, As[i].shape[1])

        ni = As[i].shape[0]
        TA[i, 0:ni, :] = As[i]

    return TA


def image_downsample_matrix(w, h):
    # [w,h] is original image size
    # 0.5[w,h] is downsampled image size

    #X = np.vstack([i*np.ones(h, dtype=np.int32) for i in range(w)])
    #Y = np.vstack([np.arange(h, dtype=np.int32) for i in range(w)])

    I = np.reshape(np.arange(w * h, dtype=np.int32), [w, h])

    adjs = np.zeros([w//2,h//2,4], dtype=np.int32)
    weights = 0.25*np.ones([w//2,h//2,4], dtype=np.float64)

    for i in range(w//2):
        for j in range(h//2):
            adjs[i,j,0] = I[2*i+0, 2*j+0]
            adjs[i,j,1] = I[2*i+0, 2*j+1]
            adjs[i,j,2] = I[2*i+1, 2*j+0]
            adjs[i,j,3] = I[2*i+1, 2*j+1]

    return np.reshape(adjs, [w//2*h//2,4]), np.reshape(weights, [w//2*h//2,4])


def batch_image_downsample_matrix(batch, w, h):

    DSI, DSW = image_downsample_matrix(w, h)

    BDSI = np.zeros([batch, w // 2 * h // 2, 4], dtype=np.int32)

    BDSW = np.zeros([batch, w // 2 * h // 2, 4], dtype=np.float64)

    for i in range(batch):
        BDSI[i,:,:] = DSI
        BDSW[i,:,:] = DSW

    return BDSI, BDSW


import sklearn.neighbors
def downsample_matrix(V0,V1,k=10):

    # parameters are chosen for mesh of scale of unit sphere [-1,+1]

    m = V0.shape[0]
    n = V1.shape[0]

    # print("downsample_matrix() m,n size: %d, %d"%( m,n) )

    # D = scipy.spatial.distance_matrix(V1,V0) # n by m matrix

    # tree = scipy.spatial.KDTree(V0)
    tree = sklearn.neighbors.KDTree(V0)

    dists, indices = tree.query(V1, k=k)

    eps = 1e-3

    weights = 1 / (dists + eps)

    weights = weights / (weights.sum(axis=1))[:, np.newaxis]

    return weights, indices


def batch_downsample_matrix(Vs, Vs_d, k=10):

    b = Vs.shape[0]

    TDSW = np.zeros([b,Vs_d.shape[1],k])
    TDSI = -np.ones([b,Vs_d.shape[1],k])

    for i in range(b):
        TDSW[i, :, :], TDSI[i, :, :] = downsample_matrix(Vs[i,:,:], Vs_d[i,:,:], k=k)

    return TDSW, TDSI

