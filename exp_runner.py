from ast import arg
import os
from tkinter.tix import Tree
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
import time
import logging
logging.basicConfig(level=logging.WARNING)
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
from models.phototourism import PhototourismDataset
from torch.utils.data import DataLoader
from models.fields import NeRF_transient, RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer_forNeuS import NeuSRenderer
import tool.visualize as vis


class Runner:
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
        self.root_dir_p = self.conf['general.root_dir_p']
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        # self.dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400))
        self.use_mask = self.conf.get_bool('train.use_mask')
        self.dataset = PhototourismDataset(root_dir=self.root_dir_p, img_downscale=1, use_mask=self.use_mask)
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
        params_to_train += list(self.embedding_a.parameters())
        params_to_train += list(self.embedding_t.parameters())
        params_to_train += list(self.transient_network.parameters())
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.transient_network,
                                     self.embedding_a,
                                     self.embedding_t,
                                     **self.conf['model.neus_renderer'])

        self.train_loader = DataLoader(self.dataset,
                                       shuffle=True,
                                       num_workers=4,
                                       batch_size=1024,
                                       pin_memory=True)

        # self.valid_dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400), split='test')
        self.valid_dataset = PhototourismDataset(root_dir=self.root_dir_p, split='val', img_downscale=2)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()

        epoch = 500

        for i in range(epoch):
            loop = tqdm(self.train_loader)
            for all_rays, rgbs in loop:
                # data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
                true_rgb = rgbs.to(self.device)
                rays = all_rays.to(self.device)
                rays_o, rays_d, near, far = rays[:, :3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
                ts = rays[:, 8].to(torch.long)
                if self.use_mask:
                    mask = rays[:, 9:10].to(torch.long)
                    # select_mask = rays[:, 10:11]
                else:
                    mask = torch.ones_like(near)
                    # select_mask = torch.zeros_like(mask)
                '''rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)'''

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3]).to(self.device)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask)

                # select_mask = (select_mask > 0.5)

                mask_sum = mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                background_rgb=background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                ts=ts,
                                                test_time=False)

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                if 'transient_sigmas' in render_out:
                    transient_sigmas = render_out['transient_sigmas']
                    betas = render_out['betas']

                # Loss
                if 'transient_sigmas' in render_out:
                    color_error = ((color_fine - true_rgb) * mask) / (betas.unsqueeze(1)**2)
                    beta_loss = 3 + torch.log(betas).mean()
                    sigma_loss = transient_sigmas.mean()
                else:
                    color_error = ((color_fine - true_rgb) * mask)
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error
                
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

                if 'transient_sigmas' in render_out:
                    loss = color_fine_loss +\
                        eikonal_loss * self.igr_weight +\
                        mask_loss * self.mask_weight +\
                        beta_loss +\
                        sigma_loss * 0.01
                else:
                    loss = color_fine_loss +\
                        eikonal_loss * self.igr_weight +\
                        mask_loss * self.mask_weight
            
                postfix = {'epoch': i, 'loss':loss.item(), 'c_l':color_fine_loss.item(), 'n_l': eikonal_loss.item(), 'psnr':psnr.item()}
                loop.set_postfix(postfix)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0 or self.iter_step == 1:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == 1:
                    self.validate_mesh()

                self.update_learning_rate()

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
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.transient_network.load_state_dict(checkpoint['transient'])
        self.embedding_a.load_state_dict(checkpoint['a'])
        self.embedding_t.load_state_dict(checkpoint['t'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'a': self.embedding_a.state_dict(),
            't': self.embedding_t.state_dict(),
            'transient': self.transient_network.state_dict(),
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=[0]):
        results = {}
        results['idx'] = idx
        results['img_gt'] = []
        results['img_fine'] = []
        for i in idx:
            sample = self.valid_dataset[i]
            rays = sample['rays'].to(self.device)
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            img_gt = sample['rgbs']
            ts = rays[:, 8].to(torch.long)
            W, H = sample['img_wh']
            img_gt = img_gt.reshape(H, W, 3)
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            ts = ts.reshape(-1).split(self.batch_size)

            out_rgb_fine = []
            weight = []
            depth = []

            for rays_o_batch, rays_d_batch, ts_batch in zip(rays_o, rays_d, ts):
                batch = rays_o_batch.shape[0]
                near = rays[:batch, 6:7]
                far = rays[:batch, 7:8]
                background_rgb = torch.ones([1, 3]).to(self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb,
                                                ts=ts_batch,
                                                test_time=True)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                if feasible('weight_sum'):
                    weight.append(render_out['weight_sum'].detach().cpu().numpy())
                if feasible('depth'):
                    depth.append(render_out['depth'].detach().cpu().numpy())
                '''if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy()
                    out_normal_fine.append(normals)
                del render_out'''

            if len(out_rgb_fine) > 0:
                img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
                results['img_fine'].append(img_fine)
                img_fine = (img_fine * 255).clip(0, 255)
            if len(weight) > 0:
                img_weight = (np.concatenate(weight, axis=0).reshape([H, W, 1]))
                x = img_weight
                mi = np.min(x) # get minimum depth
                ma = np.max(x)
                x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
                x = (255 * x).astype(np.uint8)
                img_weight = np.repeat(x, 3, axis=2)
            if len(depth) > 0:
                img_depth = (np.concatenate(depth, axis=0).reshape([H, W, 1]))
                x = img_depth
                mi = np.min(x) # get minimum depth
                ma = np.max(x)
                x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
                x = (255 * x).astype(np.uint8)
                img_depth = np.repeat(x, 3, axis=2)
            # change bgr 2 rgb
            img_fine = cv.cvtColor(img_fine, cv.COLOR_BGR2RGB)
            
            '''if len(out_normal_fine) > 0:
                normal_img = np.concatenate(out_normal_fine, axis=0)
                rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
                normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)'''

            # os.makedirs(os.path.join(self.base_exp_dir, 'distill'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
            # cv.imwrite(os.path.join(self.base_exp_dir, 'distill', name + '_distill.jpg'), img_fine)
            cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{:d}.png'.format(self.iter_step, i)), img_fine)
            # cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_depth.png'.format(self.iter_step)), img_depth)
            results['img_gt'].append(img_gt)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

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

            out_rgb_fine = []
            weight = []
            depth = []
            visibility = []
            out_normal_fine = []

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
                                                test_time=True, compute_visibility=True)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                if feasible('depth'):
                    depth.append(render_out['depth'].detach().cpu().numpy())
                if feasible('visibility'):
                    visibility.append(render_out['visibility'].detach().cpu().numpy())
                '''if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy()
                    out_normal_fine.append(normals)
                del render_out'''

            img_fine = None
            img_depth = None
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            if len(depth) > 0:
                img_depth = (np.concatenate(depth, axis=0).reshape([H, W, 1]))
                x = img_depth
                mi = np.min(x) # get minimum depth
                ma = np.max(x)
                x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
                x = (255 * x).astype(np.uint8)
                img_depth = np.repeat(x, 3, axis=2)
            if len(visibility) > 0:
                visi = (np.concatenate(depth, axis=0).reshape([H * W, -1]))
                if i == 0: # check
                    os.makedirs(os.path.join(self.base_exp_dir, 'surface'), exist_ok=True)
                    vis.visualize_visibility(visi.reshape(H, W, -1).numpy(), 
                                             os.path.join(self.base_exp_dir, 'surface', 'vis.mp4'))

            # change bgr 2 rgb
            r = img_fine[..., 2:3]
            g = img_fine[..., 1:2]
            b = img_fine[..., 0:1]
            img_fine = np.concatenate([r, g, b], axis=-1)
            
            normal_img = None
            '''if len(out_normal_fine) > 0:
                normal_img = np.concatenate(out_normal_fine, axis=0)
                rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
                normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)'''

            os.makedirs(os.path.join(self.base_exp_dir, 'surface'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'surface', 'visibility'), exist_ok=True)
            output_path = os.path.join(self.base_exp_dir, 'surface/visibility', image_name + '.npz')
            np.savez_compressed(output_path, visibility=visibility)
            cv.imwrite(os.path.join(self.base_exp_dir, 'surface', image_name + '.png'.format(self.iter_step)), img_fine)
            cv.imwrite(os.path.join(self.base_exp_dir, 'surface', image_name + '_depth.png'.format(self.iter_step)), img_depth)
            print(self.iter_step)


if __name__ == '__main__':
    print('Hello Wooden')

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
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=2048, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'validate_visibility':
        with torch.no_grad():
            runner.validate_visibility()
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
