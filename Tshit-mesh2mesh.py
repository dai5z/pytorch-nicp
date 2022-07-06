# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

from random import randint
import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh
from pytorch3d.ops import knn_gather,knn_points
# demo for registering mesh
# estimate landmark for target meshes
# the face must face toward z axis
# the mesh or point cloud must be normalized with normalize_mesh/normalize_pcl function before feed into the nicp process
device = torch.device('cuda:0')
meshes = io3d.load_obj_as_mesh('./data/human.obj', device = device,load_textures=False)
cmeshes = io3d.load_obj_as_mesh('./data/zfq_pants.obj',device = device,load_textures=False)
'''
with torch.no_grad():
    norm_meshes, norm_param = normalize_mesh(meshes)
    norm_Tmeshes,Tnorm_param = normalize_mesh(Tmeshes)
'''    
'''
    dummy_render = render.create_dummy_render([1, 0, 0], device = device)
    target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render)
    T_lm_index,T_lm_mask = get_mesh_landmark(norm_Tmeshes,dummy_render)
    #bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'))
    lm_mask = torch.all(lm_mask, dim = 0)
    T_lm_mash = torch.all(T_lm_mask,dim = 0)
    #bfm_lm_index_m = bfm_lm_index[:, lm_mask]
    target_lm_index_m = target_lm_index[:, lm_mask]
    T_lm_index_m = T_lm_index[:,T_lm_mask]
'''
c_verts = cmeshes.verts_padded()
verts = meshes.verts_padded()
lm = []
clm = []
knn = knn_points(c_verts,verts)
tmp = knn.idx.reshape(knn.idx.shape[1])
while 1:
    i = randint(0,c_verts.shape[1])
    if i not in clm:
        clm.append(i)
        lm.append(tmp[i].tolist())
    if len(clm) == 100:
        break
clm=torch.from_numpy(np.array([clm])).to(device)
lm =torch.from_numpy(np.array([lm])).to(device)

fine_config = json.load(open('config/fine_grain.json'))
registered_mesh = non_rigid_icp_mesh2mesh(cmeshes, meshes, clm, lm, fine_config)
io3d.save_meshes_as_objs(['data/result_auto_lm/fpants.obj'], registered_mesh, save_textures = False)
print('ok')