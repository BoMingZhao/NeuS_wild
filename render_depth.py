import trimesh 
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
from models.phototourism import PhototourismDataset
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def fix_pose(pose):
    # 3D Rotation about the x-axis.
    t = np.pi
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R
    return pose @ axis_transform

def fix_pose_(pose):
    # 3D Rotation about the x-axis.
    R = np.array([[0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R.T
    return pose @ axis_transform


def read_meta(root_dir_p):
    name_list = []
    dataset = PhototourismDataset(root_dir=root_dir_p, img_downscale=1, use_mask=True)
    poses = dataset.poses
    # poses[..., 3] *= 2
    id_list = dataset.img_ids
    K = dataset.Ks[id_list[0]]
    for i in range(poses.shape[0]):
        name_list.append(dataset.image_paths[id_list[i]].replace('/', '_'))

    return poses, name_list, K

def render(mesh, pose, K):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    intrinsics = K

    width, height = 1024, 576
    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene()
    scene.add(mesh)
    cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                    fx=intrinsics[0, 0], fy=intrinsics[1, 1])
    scene.add(cam, pose=fix_pose(pose))

    color, depth =  renderer.render(scene)

    return color, depth

if __name__ == '__main__':
    root_dir_p = '/data1/zhaoboming/Cambridge/GreatCourt/'
    mesh_path = '/home/zhaoboming/code/Neus_wild/exp/greatcourt/pretrain_Neus/meshes/00300000.ply'
    # mesh_path = '/home/zhaoboming/NeuS/exp/KingsCollege/pretrain_Neus/meshes/00110000.ply'
    output_dir = os.path.join(root_dir_p, 'depth_render_neus')
    os.makedirs(output_dir, exist_ok=True)


    mesh = trimesh.load(mesh_path) 
    poses, name_list, K = read_meta(root_dir_p)
    for i, name in enumerate(tqdm(name_list)):
        seq = name.split('_')[0]
        output_name = name.split('_')[1]
        os.makedirs(os.path.join(output_dir, seq), exist_ok=True)
        c2w = poses[i]
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4]) 
        c2w = np.concatenate([c2w, bottom], axis=0)
        # w2c = np.linalg.inv(c2w)
        color, depth = render(mesh, c2w, K)
        '''mi = np.min(depth)
        ma = np.max(depth)
        depth = (depth - mi) / max(ma - mi, 1e-8)
        depth = (255 * depth).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)'''
        # cv2.imwrite(os.path.join(output_dir, seq, output_name), depth)
        cv2.imwrite(os.path.join(output_dir, seq, output_name), (depth * 1000.0).astype(np.uint16))