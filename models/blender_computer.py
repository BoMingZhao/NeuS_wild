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
class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False, use_mask=False):
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
        self.object_bbox_min = np.array([-1.1, -1., 0.4])
        self.object_bbox_max = np.array([1.2, 0.5, 1.8])
        self.n_images = 250
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        self.near = 5.0
        self.far = 11.0
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
            id = 1
            c2w = np.loadtxt(f"/mnt/nas_8/group/liruoyang/blender_data/pose/{id:d}.txt")
            blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            c2w = c2w @ blender2opencv
            c2w = torch.from_numpy(c2w)
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                self.all_sky_percent = []
                for id_ in self.img_ids_train:
                    if self.use_daytime_dataset:
                        if id_ not in self.daytime_ids:
                            continue
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    if self.use_mask:
                        id = 0
                        mask = Image.open(os.path.join(self.root_dir, 'mask', f'{id:d}.png')).convert('RGB')
                    else:
                        mask = img

                    img_w, img_h = img.size
                    K = torch.from_numpy(np.loadtxt('/mnt/nas_8/group/liruoyang/blender_data/intrinsic.txt').astype(np.float))
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                        mask = mask.resize((img_w, img_h), Image.NEAREST)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    mask = self.transform(mask)
                    mask = mask.view(3, -1).permute(1, 0)[:, 0:1] # (h*w, 1) RGB
                    mask_sky_all = mask.sum().item()
                    sky_percent = 1 - (mask_sky_all / (img_h * img_w))
                    self.all_sky_percent.append(sky_percent)
                    self.all_rgbs += [img]
                    
                    directions = get_ray_directions(img_h, img_w, K)
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = torch.ones(len(rays_o), 1) # all 1

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.near * torch.ones_like(rays_o[:, :1]),
                                                self.far * torch.ones_like(rays_o[:, :1]),
                                                rays_t, mask],
                                                1)] # (h*w, 10)   
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
                self.all_sky_percent = torch.tensor(self.all_sky_percent)
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            return self.all_rays[idx, :11], self.all_rgbs[idx]

        elif self.split in ['val', 'test_train']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            ts = id_ * torch.ones([len(rays_o), 1], dtype=torch.long)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1]), ts],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        else:
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
