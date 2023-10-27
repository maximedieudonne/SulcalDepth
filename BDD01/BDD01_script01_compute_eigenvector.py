import BDD01.configuration as cfg
import slam.differential_geometry as sdg
import slam.io as sio
import os
import numpy as np



mesh_path = cfg.MESH_FOLDER
mesh = sio.load_mesh(mesh_path)
Nbv = len(mesh.vertices)
N0 = 0
N = 5
eig_vect = sdg.mesh_laplacian_eigenvectors(mesh, nb_vectors=100)
eig_vect = np.transpose(eig_vect)
L,B = sdg.compute_mesh_laplacian(mesh)
print('product')
V = np.array(mesh.vertices.T)
Pp0 = V @ B
Pp = Pp0 @ np.transpose(eig_vect[N0:N])

new_vertx = np.sum(Pp[0] * np.transpose(eig_vect[N0:N]), axis=1)
new_verty = np.sum(Pp[1] * np.transpose(eig_vect[N0:N]), axis=1)
new_vertz = np.sum(Pp[2] * np.transpose(eig_vect[N0:N]), axis=1)

new_vert = np.transpose(np.array([new_vertx,new_verty,new_vertz]))
#new_mesh = trimesh.Trimesh(vertices = new_vert, faces=mesh.faces, edges = mesh.edges)
new_mesh = mesh
new_mesh.vertices = new_vert
#sio.write_texture(stex.TextureND(darray=eig_vect), os.path.join(wd, 'eigen_vect.gii'))
fname  = 'eigmesh'
sio.write_mesh(new_mesh, os.path.join(cfg.SAVE_FOLDER, fname + str(N) + '.gii'))








