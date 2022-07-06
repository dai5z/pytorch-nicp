from cmath import e
from os import device_encoding
import torch
import io3d
import torch
import numpy as np
from utils import mesh_boundary
device = torch.device('cuda:0')
'''meshes = io3d.load_obj_as_mesh('./finalT.obj', device = device,load_textures=False)
verts = meshes.verts_packed()
vert_normals = meshes.verts_normals_packed()
boundary = torch.logical_not(mesh_boundary(meshes.faces_packed(),verts.shape[0])).float().reshape(9737,1)
k=0.002
verts0 = verts.add(vert_normals*k)
verts1 = verts.add(vert_normals*k*boundary)
meshes0 = meshes.update_padded(torch.from_numpy(np.array([verts0.tolist()])).to(device))
meshes1 = meshes.update_padded(torch.from_numpy(np.array([verts1.tolist()])).to(device))
io3d.save_meshes_as_objs(['finalT0.obj'],meshes0,save_textures=False)
io3d.save_meshes_as_objs(['finalT1.obj'],meshes1,save_textures=False)'''

meshes_out = io3d.load_obj_as_mesh('./finalT_not_norm.obj',device=device,load_textures=False)
meshes_in = io3d.load_obj_as_mesh('./test_data/Tshirt_v0.obj',device=device,load_textures=False)
verts_out = meshes_out.verts_packed()
verts_in = meshes_in.verts_packed()
arr = []
print(verts_in[360])
for i,vert in enumerate(verts_in):
    if vert[2]>0.4244:
        arr.append((vert*(0.51-vert[2])/(0.51-0.4244)+verts_out[i]*(vert[2]-0.4244)/(0.51-0.4244)).tolist())
    elif vert[2]>0.51:
        arr.append(verts_out[i].tolist())
    else:
        arr.append(vert.tolist())
verts = torch.from_numpy(np.array(arr)).to(device)
mesh = meshes_in.update_padded(torch.from_numpy(np.array([verts.tolist()])).to(device))
io3d.save_meshes_as_objs(['finalT_mix.obj'],mesh,save_textures=False)





