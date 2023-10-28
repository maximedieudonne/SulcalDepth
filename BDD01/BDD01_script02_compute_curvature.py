import BDD01.configuration as cfg
import slam.differential_geometry as sdg
import slam.curvature as scurv
import slam.texture as stex
import slam.io as sio
import os
import numpy as np
import pickle

mesh_path = cfg.MESH_FOLDER
mesh = sio.load_mesh(mesh_path)

#load curvature and laplacien
with open(os.path.join(cfg.SAVE_FOLDER, 'L.pkl'), 'rb') as file:
    L = pickle.load(file)
with open(os.path.join(cfg.SAVE_FOLDER, 'B.pkl'), 'rb') as file:
    B = pickle.load(file)

curv = sio.load_texture(os.path.join(cfg.SAVE_FOLDER, 'curv.gii'))
curv = curv.darray[0]
## analyse synthese

Nbv = len(mesh.vertices)
N0 = 0
N=500


eig_vect = sdg.mesh_laplacian_eigenvectors(mesh, nb_vectors=N)


print('product')
C =  np.array(curv)
Pp0 = C @ B

for nn in np.arange(N0,N):
    coefF = Pp0 @ eig_vect[:, N0:nn]
    Pp = coefF @ np.transpose(eig_vect[:, N0:nn])
    fname  = 'eigcurv_'
    sio.write_texture(stex.TextureND(darray=Pp), os.path.join(cfg.SAVE_FOLDER, fname + str(nn) + '.gii'))







