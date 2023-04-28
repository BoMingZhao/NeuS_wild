from select import select
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

l = [6,9,10,12,14,17,18,20,21,25,26,27,28,29,31,33,35,36,44,48,50,55,57,62,70,74,77,82,84,87,88,90,91,92,94,96,98,99,102,105,106,107,108,
112,114,115,120,123,124,125,128,129,130,134,136,138,140,142,143,149,152,154,155,156,157,158,159,160,161,162,166,168,171,172,173,177,
180,181,182,190,192,194,195,196,197,200,201,205,207,210,211,221,227,229,240,241,248,249,250,253,254,255,268,270,271,272,275,277,278,279,
283,284,287,289,290,291,292,294,296,297,298,303,304,307,310,311,312,316,318,319,321,322,327,329,332,333,335,346,351,353,354,356,357,359,
360,361]
overfit = [6, 9, 10, 14, 17]
class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='adapation', img_downscale=1, val_num=1, use_cache=False, 
                 use_mask=False, fitting_id=0):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.use_mask = use_mask
        self.fitting_id = fitting_id
        self.n_images = 1
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_adapation = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='adapation']
        self.N_images_adapation = len(self.img_ids_adapation)
        if self.split == 'fitting':
            self.all_rays = []
            self.all_rgbs = []
            id_ = self.img_ids_adapation[self.fitting_id]

            c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                            self.image_paths[id_])).convert('RGB')
            if self.use_mask:
                mask = Image.open(os.path.join(self.root_dir, 'dense/human_mask',
                                            self.image_paths[id_]).replace('.jpg', '_seg.png')).convert('RGB')
            else:
                mask = img

            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
                mask = mask.resize((img_w, img_h), Image.NEAREST)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            mask = self.transform(mask)
            mask = mask.view(3, -1).permute(1, 0)[:, 0:1] # (h*w, 1) RGB
            self.all_rgbs += [img]
            
            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays_t = id_ * torch.ones(len(rays_o), 1)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                        self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                        self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                        rays_t, mask],
                                        1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'adapation':
            return self.N_images_adapation
        elif self.split == 'fitting':
            return len(self.all_rays)

    def __getitem__(self, idx):   
        if self.split == 'fitting': # use data in the buffers
            return self.all_rays[idx, :14], self.all_rgbs[idx]
        
        elif self.split == 'adapation':
            id_ = self.img_ids_adapation[idx]
            print('id: ', id_)
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])
            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img
            sample['img_wh_ori'] = torch.LongTensor([img_w, img_h])
            img_w *= self.scale[0]
            img_h *= self.scale[1]
            K = self.Ks[id_]
            K[0, 2] *= self.scale[0] # cx
            K[1, 2] *= self.scale[1] # cy
            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            ts_id = id_
            ts_name = self.image_paths[ts_id].replace(".jpg", '')
            ts = ts_id * torch.ones([len(rays_o), 1], dtype=torch.long)
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1]), ts],
                              1)
            sample['rays'] = rays
            sample['name'] = ts_name
            sample['ts'] = ts
            sample['img_wh'] = torch.LongTensor([img_w, img_h])


        return sample
