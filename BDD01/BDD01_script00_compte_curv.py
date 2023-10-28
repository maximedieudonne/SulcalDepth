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

# compute
print("compute laplacian")
L,B = sdg.compute_mesh_laplacian(mesh)

with open(os.path.join(cfg.SAVE_FOLDER, 'L.pkl'), 'wb') as file:
    pickle.dump(L, file)

with open(os.path.join(cfg.SAVE_FOLDER, 'B.pkl'), 'wb') as file:
    pickle.dump(B, file)

print('compute curvature')
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
K1 = PrincipalCurvatures[0, :]
K2 = PrincipalCurvatures[1, :]
curv = 0.5 * (K1 + K2)
fname_curv  = 'curv'

sio.write_texture(stex.TextureND(darray=curv), os.path.join(cfg.SAVE_FOLDER, 'curv.gii'))
