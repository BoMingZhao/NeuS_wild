# Train final pipeline

import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from cv2 import repeat
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import imageio
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
from models.fields import NeRF_transient, RenderingNetwork, SDFNetwork, SingleVarianceNetwork, \
                          NeRF, NeRF_visibility, Brdf, Sky, affine_net, shadow, sky_generate
from models.LavalAE.model.Autoencoder import SkyDecoder
from models.renderer import NeuSRenderer
import tool.visualize as vis
from einops import repeat, rearrange, reduce

logging.getLogger('PIL').setLevel(logging.WARNING)

class relit_runner:
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
        self.pretrain_geometry_dir = self.conf['general.geometry_exp_dir']
        self.pretrain_visibility_dir = self.conf['general.visibility_exp_dir']
        self.sky_exp_dir = self.conf['general.sky_exp_dir']
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'])
        self.root_dir_p = self.conf['general.root_dir_p']
        self.root_dir_distill = self.conf['general.root_dir_distill']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        # self.dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400))
        self.use_mask = self.conf.get_bool('train.use_mask')
        self.use_daytime_dataset = self.conf.get_bool('train.use_daytime_dataset')
        self.overfitting = self.conf.get_bool('train.overfit')
        self.use_transient = self.conf.get_bool('train.use_transient')
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
        self.distill_weight = self.conf.get_float('train.distill_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.affine_weight = self.conf.get_float('train.affine_weight')
        self.shadow_weight = self.conf.get_float('train.shadow_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.use_affine = self.conf.get_bool('train.use_affine')
        self.use_shadow = self.conf.get_bool('train.use_shadow')
        self.use_sky_generate = self.conf.get_bool('train.use_sky_generate')
        self.embedding_a = torch.nn.Embedding(3200, 64).to(self.device)
        if self.use_transient:
            self.embedding_t = torch.nn.Embedding(3200, 16).to(self.device)
            self.transient_network = NeRF_transient(**self.conf['model.transient']).to(self.device)
            params_to_train += list(self.embedding_t.parameters())
            params_to_train += list(self.transient_network.parameters())
        else:
            self.embedding_t = None
            self.transient_network = None
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.brdf_network = Brdf(**self.conf['model.brdf']).to(self.device)
        self.sky_decoder = SkyDecoder(**self.conf['model.sky']).to(self.device)
        if self.use_shadow:
            self.shadow_net = shadow(**self.conf['model.shadow']).to(self.device)
            params_to_train += list(self.shadow_net.parameters())
            self.visibility_network = None
        else:
            self.visibility_network = NeRF_visibility(**self.conf['model.visibility']).to(self.device)
            self.shadow_net = None
        if self.use_affine:
            self.affine = affine_net(**self.conf['model.affine_net']).to(self.device)
            self.embedding_affine = torch.nn.Embedding(3200, 64).to(self.device)
            params_to_train += list(self.affine.parameters())
            params_to_train += list(self.embedding_affine.parameters())
        else:
            self.embedding_affine = None
            self.affine = None
        if self.use_sky_generate:
            self.sky_generate_net = sky_generate(**self.conf['model.sky_generate']).to(self.device)
            self.embedding_generate = torch.nn.Embedding(3200, 64).to(self.device)
            params_to_train += list(self.sky_generate_net.parameters())
            params_to_train += list(self.embedding_generate.parameters())
        else:
            self.sky_generate_net = None
            self.embedding_generate = None
        params_to_train += list(self.brdf_network.parameters())
        params_to_train += list(self.embedding_a.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.transient_network,
                                     self.embedding_a,
                                     self.embedding_t,
                                     visibility_network=self.visibility_network,
                                     brdf_network=self.brdf_network,
                                     # sky_decoder=self.sky_decoder_test,
                                     sky_decoder=self.sky_decoder,
                                     affine = self.affine,
                                     embedding_affine = self.embedding_affine,
                                     shadow_net = self.shadow_net,
                                     sky_generate_net=self.sky_generate_net,
                                     embedding_generate=self.embedding_generate,
                                     **self.conf['model.neus_renderer'])
        if mode == 'train':
            self.dataset = PhototourismDataset(root_dir=self.root_dir_p, distill_dir=self.root_dir_distill, img_downscale=2, use_mask=self.use_mask,
                                               use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting)
            print('dataset size: ', len(self.dataset.img_ids_train))

            self.train_loader = DataLoader(self.dataset,
                                           shuffle=True,
                                           num_workers=4,
                                           batch_size=self.batch_size,
                                           pin_memory=True)

        # self.valid_dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400), split='test')
        self.valid_dataset = PhototourismDataset(root_dir=self.root_dir_p, split='test', 
                                                use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting)
                                                # use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting, img_downscale=4)

        # Load geometry checkpoint
        latest_model_name = None
        model_list_raw = os.listdir(os.path.join(self.pretrain_geometry_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        logging.info('Find geometry checkpoint: {}'.format(latest_model_name))
        self.load_geometry_checkpoint(latest_model_name)
        
        # Load visibility checkpoint
        if not self.use_shadow:
            v_name = None
            model_list_raw = os.listdir(os.path.join(self.pretrain_visibility_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            v_name = model_list[-1]
            logging.info('Find visibility checkpoint: {}'.format(v_name))
            self.load_v_checkpoint(v_name)

        # load sky decoder
        logging.info('Find sky checkpoint: {}'.format(self.sky_exp_dir))
        self.load_sky_checkpoint(self.sky_exp_dir)

        # Load brdf checkpoint
        relit_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            relit_name = model_list[-1]
            
        if relit_name is not None:
            logging.info('Find relit checkpoint: {}'.format(relit_name))
            self.load_relit_checkpoint(relit_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        image_perm = self.get_image_perm()

        epoch = 5000

        for i in range(epoch):
            loop = tqdm(self.train_loader)
            for all_rays, rgbs in loop:
                # data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
                true_rgb = rgbs.to(self.device)
                rays = all_rays.to(self.device)
                rays_o, rays_d, near, far = rays[:, :3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
                ts = rays[:, 8].to(torch.long)
                distill_rgb = rays[:, 11:14]
                '''rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)'''

                background_rgb = None
                if self.use_mask:
                    mask = rays[:, 9:10].to(torch.long)
                else:
                    mask = torch.ones_like(near)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5

                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                background_rgb=background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                ts=ts,
                                                test_time=False,
                                                use_transient=self.use_transient,
                                                relit=True)

                color_fine = render_out['color_fine']
                color_static = render_out['color_static']
                if 'transient_sigmas' in render_out:
                    transient_sigmas = render_out['transient_sigmas']
                    betas = render_out['betas']

                if 'transient_sigmas' in render_out:
                    color_error = ((color_fine - true_rgb) * mask) / (betas.unsqueeze(1)**2)
                    distill_error = ((color_static - distill_rgb) * mask)
                    beta_loss = 3 + torch.log(betas).mean()
                    sigma_loss = transient_sigmas.mean()
                else:
                    color_error = ((color_fine - true_rgb) * mask)
                    distill_error = ((color_static - distill_rgb) * mask)

                if 'affine' in render_out and self.affine_weight > 0.:
                    affine_matrix = render_out['affine']
                    if affine_matrix.shape[1] == 12:
                        affine_matrix = render_out['affine'].reshape([-1, 3, 4])
                        offset = affine_matrix[:, :, 3:4].reshape(-1, 3)
                        offset_zero = torch.zeros_like(offset)
                        offset_error = F.mse_loss(offset, offset_zero, reduction='mean')
                        transpose = affine_matrix[:, :, :3].reshape(-1, 3, 3)
                        # transpose_eye = repeat(torch.eye(3), 'c1 c2 -> n c1 c2', n=transpose.shape[0]).to(transpose.device)
                        transpose_zero = torch.zeros_like(transpose)
                        transpose_error = F.mse_loss(transpose, transpose_zero, reduction='mean')
                        affine_loss = transpose_error + offset_error
                    else:
                        affine_matrix = render_out['affine'].reshape([-1, 6])
                        offset = affine_matrix[:, 3:6].reshape(-1, 3)
                        offset_zero = torch.zeros_like(offset)
                        offset_error = F.mse_loss(torch.abs(offset), offset_zero, reduction='mean')
                        scale = affine_matrix[:, 0:3].reshape(-1, 3)
                        scale_one = torch.ones_like(scale)
                        scale_error = F.mse_loss(torch.abs(scale), scale_one, reduction='mean')
                        affine_loss = scale_error + offset_error
                else:
                    affine_loss = 0

                if 'v_surface' in render_out:
                    # v_surface = torch.abs(render_out['v_surface'])
                    # v_1 = torch.ones_like(v_surface)
                    # shadow_loss = F.mse_loss(v_surface, v_1, reduction='mean')
                    shade = render_out['shade']
                    shadow_loss = F.mse_loss(shade, torch.ones_like(shade), reduction='mean')
                else:
                    shadow_loss = 0

                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                distill_loss = F.mse_loss(distill_error, torch.zeros_like(distill_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())


                if 'transient_sigmas' in render_out:
                    loss = color_fine_loss +\
                        distill_loss * self.distill_weight +\
                        beta_loss +\
                        sigma_loss * 4
                else:
                    # loss = color_fine_loss + distill_loss * self.distill_weight + affine_loss * self.affine_weight
                    loss = distill_loss * self.distill_weight + affine_loss * self.affine_weight + shadow_loss * self.shadow_weight
                # loss = 0.5 * ((v_pred - v_gt)**2).mean()
            
                # postfix = {'epoch': i, 'loss':loss.item(), 'c_l': color_fine_loss.item(), 'd_l': distill_loss.item() * self.distill_weight, 'psnr': psnr.item()}
                postfix = {'epoch': i, 'loss':loss.item(), 'd_l': distill_loss.item() * self.distill_weight, 'psnr': psnr.item()}

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.use_shadow:
                    postfix['s_l'] = shadow_loss.item() * self.shadow_weight
                    self.writer.add_scalar('Loss/shadow_loss', shadow_loss.item() * self.shadow_weight, self.iter_step)
                if self.use_affine:
                    self.writer.add_scalar('Loss/affine_loss', affine_loss.item() * self.affine_weight, self.iter_step)
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/distill_loss', distill_loss.item() * self.distill_weight, self.iter_step)
                loop.set_postfix(postfix)
                
                self.iter_step += 1

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                # TODO:
                if self.iter_step % self.val_freq == 0:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        results = self.validate_image()
                        for i in range(len(results['idx'])):
                            idx = results['idx'][i]
                            img_fine = results['img_fine'][i]
                            img_fine = np.transpose(img_fine, (2, 0, 1))
                            img_gt = results['img_gt'][i]
                            img_gt = np.transpose(img_gt, (2, 0, 1))
                            albedo = results['albedo'][i]
                            albedo = np.transpose(albedo, (2, 0, 1))
                            if self.use_shadow:
                                shadow_img = results['shadow'][i]
                                shadow_img = np.transpose(shadow_img, (2, 0, 1))
                                self.writer.add_image(f'shadow_{idx:d}', shadow_img, self.iter_step)

                            hdr = results['hdr'][i]
                            hdr_min = np.min(hdr)
                            hdr_max = np.max(hdr)
                            hdr = (hdr - hdr_min) / (hdr_max - hdr_min + 1e-8) # normalize to 0~1
                            hdr = (255 * hdr).astype(np.uint8)
                            hdr = np.transpose(hdr, (2, 0, 1))

                            self.writer.add_image(f'img_fine_{idx:d}', img_fine, self.iter_step)
                            self.writer.add_image(f'img_gt{idx:d}', img_gt, self.iter_step)
                            self.writer.add_image(f'albedo_{idx:d}', albedo, self.iter_step)
                            self.writer.add_image(f'hdr_{idx:d}', hdr, self.iter_step)
                    torch.cuda.empty_cache()
                
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

    def load_geometry_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.pretrain_geometry_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])

        logging.info('End')

    def load_v_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.pretrain_visibility_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.visibility_network.load_state_dict(checkpoint['visibility'])
        
        logging.info('End')

    def load_sky_checkpoint(self, checkpoint_name):
        self.sky_decoder.load_state_dict(torch.load(checkpoint_name))

        logging.info('End')

    def load_relit_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.brdf_network.load_state_dict(checkpoint['brdf'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.embedding_a.load_state_dict(checkpoint['a'])
        self.iter_step = checkpoint['iter_step']
        if self.use_transient:
            self.transient_network.load_state_dict(checkpoint['transient'])
            self.embedding_t.load_state_dict(checkpoint['t'])
        if self.use_affine:
            self.affine.load_state_dict(checkpoint['affine'])
            self.embedding_affine.load_state_dict(checkpoint['embedding_affine'])
        if self.use_shadow:
            self.shadow_net.load_state_dict(checkpoint['shadow'])
        if self.use_sky_generate:
            self.sky_generate_net.load_state_dict(checkpoint['sky_generate'])
            self.embedding_generate.load_state_dict(checkpoint['embedding_generate'])


    def save_checkpoint(self):
        checkpoint = {
            'brdf': self.brdf_network.state_dict(),
            'a': self.embedding_a.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        if self.use_transient:
            checkpoint['transient'] = self.transient_network.state_dict()
            checkpoint['t'] = self.embedding_t.state_dict()
        
        if self.use_shadow:
            checkpoint['shadow'] = self.shadow_net.state_dict()
        
        if self.use_affine:
            checkpoint['affine'] = self.affine.state_dict()
            checkpoint['embedding_affine'] = self.embedding_affine.state_dict()

        if self.use_sky_generate:
            checkpoint['sky_generate'] = self.sky_generate_net.state_dict()
            checkpoint['embedding_generate'] = self.embedding_generate.state_dict()

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=[6, 9], hdr_path=None, ts_=None):
        print('Validate: iter: {}, picture: {}'.format(self.iter_step, idx[0]))
        results = {}
        results['idx'] = idx
        results['img_gt'] = []
        results['img_fine'] = []
        results['albedo'] = []
        results['hdr'] = []
        results['shadow'] = []
        for i in idx:
            sample = self.valid_dataset[i]
            image_name = sample['name']
            rays = sample['rays'].to(self.device)
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            img_gt = sample['rgbs']
            ts = rays[:, 8].to(torch.long)
            if ts_ != None:
                ts =  ts_ * torch.ones_like(ts)
            dec_input = ts[0:1]
            # H, W = (400, 400)
            W, H = sample['img_wh']
            img_gt = img_gt.reshape(H, W, 3)
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            ts = ts.reshape(-1).split(self.batch_size)
            hdr_name = ''
            a_embedded = self.embedding_a(dec_input)
            output = self.sky_decoder(a_embedded)
            output = output.clamp_min_(0.0)
            if hdr_path == None:
                hdr_path = ['']
            for p in hdr_path:
                if p == '':
                    hdr_tensor = None
                else:
                    hdr = imageio.imread(p, format=('HDR-FI' if p.endswith('.hdr') else 'EXR'))
                    hdr_tensor = torch.from_numpy(hdr).to(self.device).reshape([-1, 3])
                    hdr_tensor.clamp_min_(0.0)
                    hdr_name = p.split('/')[-1].replace('.hdr', '')

                visibility_pred = []
                visibility_gt = []
                shadow_output = []
                out_rgb_fine = []
                out_albedo = []
                sdf = []

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
                                                    use_transient=self.use_transient,
                                                    test_time=True, relit=True, envmap=hdr_tensor)

                    def feasible(key): return (key in render_out) and (render_out[key] is not None)

                    if feasible('color_fine'):
                        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                    if feasible('albedo'):
                        out_albedo.append(torch.pow(render_out['albedo'].detach(), 1/2.2).cpu().numpy())
                        # out_albedo.append(render_out['albedo'].detach().cpu().numpy())
                    if feasible('sdf'):
                        sdf.append(render_out['sdf'].detach().cpu().numpy())
                    if feasible('v'):
                        visibility_pred.append(render_out['v'].detach().cpu().numpy())
                    if feasible('v_gt'):
                        visibility_gt.append(render_out['v_gt'].detach().cpu().numpy())
                    if feasible('v_surface'):
                        shadow_output.append(render_out['v_surface'].detach().cpu().numpy())
                    
                    '''if feasible('gradients') and feasible('weights'):
                        n_samples = self.renderer.n_samples + self.renderer.n_importance
                        normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                        if feasible('inside_sphere'):
                            normals = normals * render_out['inside_sphere'][..., None]
                        normals = normals.sum(dim=1).detach().cpu().numpy()
                        out_normal_fine.append(normals)
                    del render_out'''

                img_fine = None
                if len(out_rgb_fine) > 0:
                    img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
                    results['img_fine'].append(img_fine)
                    img_fine = (img_fine * 255).clip(0, 255)

                albedo = None
                if len(out_albedo) > 0:
                    albedo = np.concatenate(out_albedo, axis=0).reshape([H, W, 3])
                    results['albedo'].append(albedo)
                    albedo = (albedo * 255).clip(0, 255)

                sdf_img = None
                if len(sdf) > 0:
                    sdf_img = torch.from_numpy((np.concatenate(sdf, axis=0).reshape([H, W]) * 255).clip(0, 255))
                    sdf_img = repeat(sdf_img, 'n1 n2 -> n1 n2 3')
                
                '''if len(visibility_pred) > 0:
                    visi = (np.concatenate(visibility_pred, axis=0).reshape([H * W, -1]))
                    if i == 0: # check
                        os.makedirs(os.path.join(self.base_exp_dir, 'visibility'), exist_ok=True)
                        vis.visualize_visibility(visi.reshape(H, W, -1), 
                                                os.path.join(self.base_exp_dir, 'visibility', '{:0>8d}_vis_pred.mp4'.format(self.iter_step)))
                v_img = None
                if len(visibility_pred) > 0:
                    v_img = torch.from_numpy((np.concatenate(visibility_pred, axis=0).reshape([H, W]) * 255).clip(0, 255))
                    v_img = repeat(v_img, 'n1 n2 -> n1 n2 3')

                vgt_img = None
                if len(visibility_gt) > 0:
                    vgt_img = torch.from_numpy((np.concatenate(visibility_gt, axis=0).reshape([H, W]) * 255).clip(0, 255))
                    vgt_img = repeat(vgt_img, 'n1 n2 -> n1 n2 3')'''

                shadow_img = None
                if len(shadow_output) > 0:
                    shadow_img = np.concatenate(shadow_output, axis=0).reshape([H, W, 1]).repeat(3, axis=-1)
                    results['shadow'].append(shadow_img)
                    shadow_img = (shadow_img * 255).clip(0, 255)

                r = img_fine[..., 2:3]
                g = img_fine[..., 1:2]
                b = img_fine[..., 0:1]
                img_fine = np.concatenate([r, g, b], axis=-1)
                r = albedo[..., 2:3]
                g = albedo[..., 1:2]
                b = albedo[..., 0:1]
                albedo = np.concatenate([r, g, b], axis=-1)
                ts_prefix = f'{ts_}_' if ts_ else ''
                if self.use_shadow:
                    shadow_img = cv.cvtColor(shadow_img, cv.COLOR_BGR2RGB)
                    cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{}{:0>8d}_{:d}_{:s}_shadow.png'.format(ts_prefix, self.iter_step, i, hdr_name)), shadow_img)
                os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
                cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{}{:0>8d}_{:d}_{:s}.png'.format(ts_prefix, self.iter_step, i, hdr_name)), img_fine)
                cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{}{:0>8d}_{:d}_{:s}_albedo.png'.format(ts_prefix, self.iter_step, i, hdr_name)), albedo)
                # imageio.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{:d}_{:s}_v.png'.format(self.iter_step, i, hdr_name)), v_img)
                # imageio.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{:d}_{:s}_vgt.png'.format(self.iter_step, i, hdr_name)), vgt_img)
                
                recon_image = np.transpose(output[0].cpu().detach().numpy(), (1, 2, 0))
                imageio.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{}{:0>8d}_{:d}_hdr.hdr'.format(ts_prefix, self.iter_step, i)), 
                                recon_image, format=('HDR-FI'))
                
                
                results['hdr'].append(recon_image)
                results['img_gt'].append(img_gt)

            print(self.iter_step)
        return results

if __name__ == '__main__':
    print('Training relit network')

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    hdr_list = ['./models/LavalAE/samples/olat_4_0.hdr']
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
    runner = relit_runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_image':
        with torch.no_grad():
            # runner.validate_image(idx=[1, 2], hdr_path=None, ts_=6)
            # runner.validate_image(idx=[1, 2], hdr_path=None, ts_=6)
            # runner.validate_image(idx=[1, 2], hdr_path=None, ts_=10)
            # runner.validate_image(idx=[1, 2], hdr_path=None, ts_=9)
            # runner.validate_image(idx=[9, 10], hdr_path=None, ts_=9)
            runner.validate_image(idx=[6, 9, 10], hdr_path=None, ts_=6)
            runner.validate_image(idx=[6, 9, 10], hdr_path=None, ts_=9)
            runner.validate_image(idx=[6, 9, 10], hdr_path=None, ts_=10)