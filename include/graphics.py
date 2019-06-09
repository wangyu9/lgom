import numpy as np


def show_mesh_tensor(TV,TF,TC):

    import matplotlib.pyplot as plt
    import surf_util as su

    _, ax = su.show_mesh_tensor(TV, TF, TVC=TC)

    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)
    ax.set_zlim(-.5, .5)

    plt.show()


def normal2color(N):
    assert (N.shape[1] == 3)
    C = 0.5 * (N + 1)

    return np.clip(C, 0, 1)


import matplotlib
import matplotlib.cm as cm


def value2color(u, min=0, max=1, colormap=cm.Greys_r):
    n = u.shape[0]

    lst = [u[i] for i in range(n)]

    minima = min  # min(lst)
    maxima = max  # max(lst)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

    color = np.zeros([n, 4])

    for i in range(n):
        c = mapper.to_rgba(lst[i])
        # c = c[0] # somehow need this...
        for j in range(4):
            color[i, j] = c[j]

    return color


'''
def show_mesh_mayavi(V,F,*positional_parameters, ** keyword_parameters):

    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html?highlight=tri#mayavi.mlab.triangular_mesh

    def parser(name,default_value):
        if (name in keyword_parameters):
            return keyword_parameters[name]
        else:
            return default_value

    u = parser('u', [])
    v = parser('view', [])
    colorbar = parser('colorbar', False)
    colormap = parser('colormap', 'jet')

    h = C()

    from mayavi import mlab
    n = V.shape[0]
    #u = np.ones(n)
    mlab.figure(size=(1200,1200),bgcolor=(1.,1.,1.))

    if len(u)==0:
        h.mesh = mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F, color=(0.7, 0.7, 0.7))
    else:
        vmin = parser('vmin', np.min(u))
        vmax = parser('vmax', np.max(u))
        h.mesh = mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F, scalars=u, vmin=vmin, vmax=vmax, colormap=colormap,resolution=200,representation='surface')
        # colormap='Accent',RdBu,spectral
        # all possible colormap http://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
        # some visualized here: http://matplotlib.org/users/colormaps.html
    if colorbar:
        h.bar = mlab.colorbar()
        h.bar._label_text_property.color = (0, 0, 0)
    mlab.show()
    if len(v)>0:
        mlab.view(*v)

    return h
'''


def show_mesh0(V, F, *positional_parameters, **keyword_parameters):
    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html?highlight=tri#mayavi.mlab.triangular_mesh

    def parser(name, default_value):
        if (name in keyword_parameters):
            return keyword_parameters[name]
        else:
            return default_value

    u = parser('u', [])
    v = parser('view', [])
    # color = parser('color',np.ones([F.shape[0],4]))
    color = parser('color', ['blue' for i in range(F.shape[0])])
    colorbar = parser('colorbar', False)
    colormap = parser('colormap', 'jet')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.plot(xs=V[:,0],ys=V[:,1])


    tri = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F)

    return tri  # fig


def vertex2face_uniform(F, uV):
    assert (F.shape[1] == 3)

    uF = (uV[F[:, 0], :] + uV[F[:, 1], :] + uV[F[:, 2], :]) / 3

    return uF


def show_mesh(V, F, *positional_parameters, **keyword_parameters):
    import matplotlib.pyplot as plt

    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html?highlight=tri#mayavi.mlab.triangular_mesh

    def parser(name, default_value):
        if (name in keyword_parameters):
            tentative = keyword_parameters[name]
            if tentative is None:
                return default_value
            return tentative
        else:
            return default_value

    F = F.astype(np.int32)

    u = parser('u', [])
    v = parser('view', [])
    C = parser('color', 0.4*np.ones([F.shape[0], 4]))
    VC = parser('vertex_color', [])
    colorbar = parser('colorbar', False)
    colormap = parser('colormap', 'jet')

    fig = parser('fig', None)
    # fig = parser('fig', plt.figure(figsize=(20, 10)))
    ax = parser('ax', None)

    if len(u):
        VC = value2color(u, min=np.min(u), max=np.max(u), colormap=colormap)

    if len(VC):
        C = vertex2face_uniform(F, VC)

    assert (C.shape[1] == 3 or C.shape[1] == 4)
    color = [np.ma.array(C[i], mask=0) for i in range(C.shape[0])]

    from mpl_toolkits.mplot3d import Axes3D  # somehow I need to import this to
    # see https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error

    # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(20, 10))
            # ax = fig.gca(projection='3d')
            # else:
            # ax = fig

        ax = fig.gca(projection='3d')

    # ax.plot(xs=V[:,0],ys=V[:,1])

    tri = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, )  # ,color=color)

    # tri = ax.plot_trisurf(V[:, 0]+1, V[:, 1], V[:, 2], triangles=F, )  # ,color=color)

    ax.set_aspect(1)

    tri.set_facecolors(color)

    return tri, ax  # fig


def show_mesh_old(V,F,*positional_parameters, ** keyword_parameters):

    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html?highlight=tri#mayavi.mlab.triangular_mesh

    def parser(name,default_value):
        if (name in keyword_parameters):
            return keyword_parameters[name]
        else:
            return default_value

    u = parser('u', [])
    v = parser('view', [])
    colorbar = parser('colorbar', False)
    colormap = parser('colormap', 'jet')

    import eigen_solver as es
    h = es.C()

    from mayavi import mlab
    n = V.shape[0]
    #u = np.ones(n)
    mlab.figure(size=(1200,1200),bgcolor=(1.,1.,1.))

    if len(u)==0:
        h.mesh = mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F, color=(0.7, 0.7, 0.7))
    else:
        vmin = parser('vmin', np.min(u))
        vmax = parser('vmax', np.max(u))
        h.mesh = mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F, scalars=u, vmin=vmin, vmax=vmax, colormap=colormap,resolution=200,representation='surface')
        # colormap='Accent',RdBu,spectral
        # all possible colormap http://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
        # some visualized here: http://matplotlib.org/users/colormaps.html
    if colorbar:
        h.bar = mlab.colorbar()
        h.bar._label_text_property.color = (0, 0, 0)
    mlab.show()
    if len(v)>0:
        mlab.view(*v)

    return h