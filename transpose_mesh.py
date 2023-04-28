import open3d as o3d
import numpy as np
from models.phototourism import PhototourismDataset
import ipdb

root_dir_p = '/data1/zhaoboming/Cambridge/GreatCourt/'
dataset = PhototourismDataset(root_dir=root_dir_p, img_downscale=1, use_mask=True)
ipdb.set_trace()
poses = dataset.poses
c2w = poses[0]
bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4]) 
c2w = np.concatenate([c2w, bottom], axis=0)

ply_file_path = "/home/zhaoboming/code/Neus_wild/exp/shopfacade/pretrain_Neus/meshes/00250000_transpose.ply"
mesh = o3d.io.read_triangle_mesh(ply_file_path)
vertices = np.array(mesh.vertices)
# 保存缩放后的PLY文件
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(vertices)
cloud_cam = cloud.transform(c2w)

np.savetxt('shop.txt', np.asarray(cloud_cam.points))