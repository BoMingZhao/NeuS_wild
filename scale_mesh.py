import open3d as o3d
import numpy as np
from models.phototourism import PhototourismDataset
import ipdb

root_dir_p = '/data1/zhaoboming/Cambridge/ShopFacade/'
dataset = PhototourismDataset(root_dir=root_dir_p, img_downscale=1, use_mask=True)
poses = dataset.poses
c2w = poses[0]
bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4]) 
c2w = np.concatenate([c2w, bottom], axis=0)
w2c = np.linalg.inv(c2w)

ply_file_path = "/home/zhaoboming/code/Neus_wild/exp/shopfacade/pretrain_Neus/meshes/00250000.ply"
mesh = o3d.io.read_triangle_mesh(ply_file_path)

trans = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 对顶点坐标进行缩放
# scale_factor = dataset.scale_factor
scale_factor = 1
b = np.array([1.0]).reshape([1, 1])
for i in range(len(mesh.vertices)):
    mesh.vertices[i] *= scale_factor
    mesh.vertices[i] = (np.concatenate([(mesh.vertices[i].reshape([1, 3]) @ trans).reshape([1, 3]), b], axis=-1) @ w2c)[..., :3].reshape([3])

# 保存缩放后的PLY文件
output_file_path = "/home/zhaoboming/code/Neus_wild/exp/shopfacade/pretrain_Neus/meshes/00250000_transpose.ply"
o3d.io.write_triangle_mesh(output_file_path, mesh)