# Train final pipeline

import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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

class eval_runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

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
        self.embedding_t = torch.nn.Embedding(3200, 16).to(self.device)
        self.transient_network = NeRF_transient(**self.conf['model.transient']).to(self.device)
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
        else:
            self.embedding_affine = None
            self.affine = None
        if self.use_sky_generate:
            self.sky_generate_net = sky_generate(**self.conf['model.sky_generate']).to(self.device)
            self.embedding_generate = torch.nn.Embedding(3200, 64).to(self.device)
        else:
            self.sky_generate_net = None
            self.embedding_generate = None

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
        if mode == 'eval':
            self.dataset = PhototourismDataset(root_dir=self.root_dir_p, distill_dir=self.root_dir_distill, img_downscale=4, use_mask=self.use_mask,
                                               use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting)
            self.train_loader = DataLoader(self.dataset,
                                           shuffle=True,
                                           num_workers=4,
                                           batch_size=self.batch_size,
                                           pin_memory=True)

        # self.valid_dataset = BlenderDataset(root_dir=root_dir_b, img_wh=(400, 400), split='test')
        self.valid_dataset = PhototourismDataset(root_dir=self.root_dir_p, split='test_train', 
                                                # use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting)
                                                use_daytime_dataset=self.use_daytime_dataset, overfitting=self.overfitting, img_downscale=4)

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

if __name__ == '__main__':
    print('Validation')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = eval_runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'adapation':
        runner.train()