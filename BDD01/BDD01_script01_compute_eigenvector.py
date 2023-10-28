import BDD01.configuration as cfg
import slam.differential_geometry as sdg
import slam.io as sio
import os
import numpy as np
import pickle


mesh_path = cfg.MESH_FOLDER
mesh = sio.load_mesh(mesh_path)


with open(os.path.join(cfg.SAVE_FOLDER, 'L.pkl'), 'rb') as file:
    L = pickle.load(file)
with open(os.path.join(cfg.SAVE_FOLDER, 'B.pkl'), 'rb') as file:
    B = pickle.load(file)


Nbv = len(mesh.vertices)
N0 = 0
N = 500

eig_vect = sdg.mesh_laplacian_eigenvectors(mesh, nb_vectors=N)


print('product')

V = np.array(mesh.vertices.T)

Pp0 = V @ B

for nn in np.arange(N0, N):
    Pp = Pp0 @ eig_vect[:,N0:nn]
    new_vertx = Pp[0,:] @ np.transpose(eig_vect[:,N0:nn])
    new_verty = Pp[1,:] @ np.transpose(eig_vect[:,N0:nn])
    new_vertz = Pp[2,:] @ np.transpose(eig_vect[:,N0:nn])
    new_vert = np.transpose(np.array([new_vertx,new_verty,new_vertz]))

    new_mesh = mesh
    new_mesh.vertices = new_vert
    fname  = 'eigmesh_'
    sio.write_mesh(new_mesh, os.path.join(cfg.SAVE_FOLDER, fname + str(nn) + '.gii'))








