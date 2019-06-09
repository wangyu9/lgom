
import numpy as np

def load_obj(filename):
    V = []  # vertex
    F = []  # face indexies

    fh = open(filename)
    for line in fh:
        if line[0] == '#':
            continue

        line = line.strip().split(' ')
        if line[0] == 'v':  # vertex
            V.append([float(line[i+1]) for i in range(3)])
        elif line[0] == 'f':  # face
            face = line[1:]
            for i in range(0, len(face)):
                face[i] = int(face[i].split('/')[0]) - 1
            F.append(face)

    V = np.array(V)
    F = np.array(F)

    return V, F


#from os import path
#from glob import glob
#from cStringIO import StringIO
# https://github.com/Infinidat/infi.clickhouse_orm/issues/27
from io import StringIO
def load_off(filename, no_colors=True):
    lines = open(filename).readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    assert lines[0].strip() in ['OFF', 'COFF'], 'OFF header missing'
    has_colors = lines[0].strip() == 'COFF'
    n_verts, n_faces, _ = map(int, lines[1].split())
    vertex_data = np.loadtxt(
        StringIO(''.join(lines[2:2 + n_verts])),
        dtype=np.float)
    if n_faces > 0:
        faces = np.loadtxt(StringIO(''.join(lines[2+n_verts:])), dtype=np.int)[:,1:]
    else:
        faces = None
    if has_colors:
        colors = vertex_data[:,3:].astype(np.uint8)
        vertex_data = vertex_data[:,:3]
    else:
        colors = None
    if no_colors:
        return vertex_data, faces
    else:
        return vertex_data, colors, faces


def save_as_ply(filename, V, F):
    output = \
"""ply
format ascii 1.0
comment Created by Blender 2.78 (sub 0) - www.blender.org, source file: ''
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar uint vertex_indices
end_header
""".format(V.size(0), F.size(0))

    for i in range(V.size(0)):
        output += "{} {} {}\n".format(V[i][0], V[i][1], V[i][2])

    for i in range(F.size(0)):
        output += "3 {} {} {}\n".format(F[i][0], F[i][1], F[i][2])

    text_file = open(filename, "w")
    text_file.write(output)
    text_file.close()


def save_as_obj(filename, V, F):
        output = ""

        for i in range(V.size(0)):
            if V[i].abs().sum() > 0:
                output += "v {} {} {}\n".format(V[i][0], V[i][1], V[i][2])

        for i in range(F.size(0)):
            if F[i].abs().sum() > 0:
                output += "f {} {} {}\n".format(F[i][0] + 1, F[i][1] + 1, F[i][2] + 1)

        text_file = open(filename, "w")
        text_file.write(output)
        text_file.close()


def save_obj(filename, V, F):
        output = ""

        for i in range(V.shape[0]):
            #if V[i].abs().sum() > 0:
            output += "v {} {} {}\n".format(V[i][0], V[i][1], V[i][2])

        for i in range(F.shape[0]):
            #if F[i].abs().sum() > 0:
            output += "f {} {} {}\n".format(F[i][0] + 1, F[i][1] + 1, F[i][2] + 1)

        text_file = open(filename, "w")
        text_file.write(output)
        text_file.close()