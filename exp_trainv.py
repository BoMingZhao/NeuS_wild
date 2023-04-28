# Train the visibility network

from ast import arg
import os
from tkinter.tix import Tree
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import argparse
import numpy as np
import cv2 as cv
from torch._C import dtype
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
import ipdb
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.blender import BlenderDataset
from models.phototourism import PhototourismDataset
from torch.utils.data import DataLoader
from models.fields import NeRF_transient, RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, NeRF_visibility
from models.renderer import NeuSRenderer
import tool.visualize as vis

root_dir_b = '/home/zhaoboming/NeuS/public_data/lego'
# root_dir_p = '/home/zhaoboming/nerf-w/trevi_fountain'
root_dir_p = '/mnt/nas_8/group/BBYang/relight_boming_data/trevi_fountain'

class v_runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.pretrain_geometry_dir = self.conf['general.base_exp_dir']
        # self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], 'visibility_train')
        self.base_exp_dir = os.path.join(self.conf['general.base_save_exp_dir'])
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        # self.dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400))
        self.use_mask = self.conf.get_bool('train.use_mask')
        self.use_daytime_dataset = self.conf.get_bool('train.use_daytime_dataset')
        self.dataset = PhototourismDataset(root_dir=root_dir_p, img_downscale=4, use_mask=self.use_mask,
                                           use_daytime_dataset=self.use_daytime_dataset)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.embedding_a = torch.nn.Embedding(3200, 64).to(self.device)
        self.embedding_t = torch.nn.Embedding(3200, 16).to(self.device)
        self.transient_network = NeRF_transient(**self.conf['model.transient']).to(self.device)
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.visibility_network = NeRF_visibility(**self.conf['model.visibility']).to(self.device)
        '''params_to_train += list(self.embedding_a.parameters())
        params_to_train += list(self.embedding_t.parameters())
        params_to_train += list(self.transient_network.parameters())
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())'''
        params_to_train += list(self.visibility_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.transient_network,
                                     self.embedding_a,
                                     self.embedding_t,
                                     visibility_network=self.visibility_network,
                                     **self.conf['model.neus_renderer'])

        self.train_loader = DataLoader(self.dataset,
                                       shuffle=True,
                                       num_workers=4,
                                       batch_size=self.batch_size,
                                       pin_memory=True)

        # self.valid_dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400), split='test')
        self.valid_dataset = PhototourismDataset(root_dir=root_dir_p, split='test_train', 
                                                 use_daytime_dataset=self.use_daytime_dataset)

        # Load checkpoint
        latest_model_name = None
        model_list_raw = os.listdir(os.path.join(self.pretrain_geometry_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        
        v_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            v_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if v_name is not None:
            logging.info('Find checkpoint: {}'.format(v_name))
            self.load_v_checkpoint(v_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        image_perm = self.get_image_perm()

        epoch = 500

        for i in range(epoch):
            loop = tqdm(self.train_loader)
            for all_rays, rgbs in loop:
                # data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
                true_rgb = rgbs.to(self.device)
                rays = all_rays.to(self.device)
                rays_o, rays_d, near, far = rays[:, :3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
                ts = rays[:, 8].to(torch.long)
                '''rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)'''

                background_rgb = None

                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                background_rgb=background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                ts=ts,
                                                test_time=False,
                                                train_visibility=True)

                v_pred = render_out['v_pred']
                v_gt = render_out['v_gt']

                # loss = 0.5 * ((v_pred - v_gt)**2).mean()
                loss = F.binary_cross_entropy(v_gt.clip(1e-3, 1.0 - 1e-3), v_pred)
            
                postfix = {'epoch': i, 'loss':loss.item()}
                loop.set_postfix(postfix)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.val_freq == 0:
                    self.validate_visibility()
                
                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()
    
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
    
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.pretrain_geometry_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.transient_network.load_state_dict(checkpoint['transient'])
        self.embedding_a.load_state_dict(checkpoint['a'])
        self.embedding_t.load_state_dict(checkpoint['t'])

        logging.info('End')

    def load_v_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.visibility_network.load_state_dict(checkpoint['visibility'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'visibility': self.visibility_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_visibility(self, idx=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        for i in range(len(self.dataset)):
            sample = self.valid_dataset[i]
            image_name = sample['name']
            rays = sample['rays'].to(self.device)
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            ts = rays[:, 8].to(torch.long)
            # H, W = (400, 400)
            W, H = sample['img_wh']
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            ts = ts.reshape(-1).split(self.batch_size)

            visibility_pred = []
            visibility_gt = []

            for rays_o_batch, rays_d_batch, ts_batch in zip(rays_o, rays_d, ts):
                batch = rays_o_batch.shape[0]
                near = rays[:batch, 6:7]
                far = rays[:batch, 7:8]
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb,
                                                ts=ts_batch,
                                                test_time=True, train_visibility=True)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('v_pred'):
                    visibility_pred.append(render_out['v_pred'].detach().cpu().numpy())
                if feasible('v_gt'):
                    visibility_gt.append(render_out['v_gt'].detach().cpu().numpy())
                '''if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy()
                    out_normal_fine.append(normals)
                del render_out'''

            if len(visibility_gt) > 0:
                visi = (np.concatenate(visibility_gt, axis=0).reshape([H * W, -1]))
                if i == 0: # check
                    os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train'), exist_ok=True)
                    os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train', 'visibility'), exist_ok=True)
                    vis.visualize_visibility(visi.reshape(H, W, -1), 
                                             os.path.join(self.base_exp_dir, 'visibility_train', 'visibility', '{:0>8d}_vis_gt.mp4'.format(self.iter_step)))
            
            if len(visibility_pred) > 0:
                visi = (np.concatenate(visibility_pred, axis=0).reshape([H * W, -1]))
                if i == 0: # check
                    os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train'), exist_ok=True)
                    os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train', 'visibility'), exist_ok=True)
                    vis.visualize_visibility(visi.reshape(H, W, -1), 
                                             os.path.join(self.base_exp_dir, 'visibility_train', 'visibility', '{:0>8d}_vis_pred.mp4'.format(self.iter_step)))

            os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'visibility_train', 'visibility'), exist_ok=True)
            output_path = os.path.join(self.base_exp_dir, 'visibility_train', 'visibility', image_name + '.npz')
            np.savez_compressed(output_path, visibility=visibility_pred)
            print(self.iter_step)
            break

if __name__ == '__main__':
    print('Training visiblity network')

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = v_runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_visibility':
        with torch.no_grad():
            runner.validate_visibility()