import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob

def colored_data(x, cmap="jet", d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    # print(np.min(x), np.max(x))
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:, :, :3]).astype(np.uint8)  # H, W, C

scene_list = ['OldHospital', 'GreatCourt', 'StMarysChurch', 'ShopFacade', 'KingCollege']
scene_list = ['OldHospital']
for scene in scene_list:
    input_path = f"/data1/zhaoboming/Cambridge/{scene}/segment/"
    output_dir = f"/data1/zhaoboming/Cambridge/{scene}/masks/"
    sky_dir = f"/data1/zhaoboming/Cambridge/{scene}/sky_masks/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sky_dir, exist_ok=True)
    imgs_path = []
    for seq_name in os.listdir(input_path):
        os.makedirs(os.path.join(output_dir, seq_name), exist_ok=True)
        os.makedirs(os.path.join(sky_dir, seq_name), exist_ok=True)
        imgs_path.extend(glob.glob(os.path.join(input_path, seq_name, "*_category.png")))
    for img_path in tqdm(imgs_path):
        base_name = img_path.replace('_category.png', '_seg.png')
        save_name = base_name.split('/')[-2] + '/' + base_name.split('/')[-1]
        img = cv2.imread(os.path.join(input_path, img_path), cv2.IMREAD_ANYDEPTH)
        
        '''mask = ((img != 40) & (img != 37) & (img != 0) & (img != 2) & (img != 7)).astype(np.uint8) * 255
        mask.repeat(3, axis=-1)
        out_name = os.path.join(output_dir, save_name)
        cv2.imwrite(out_name, mask)'''

        mask = ((img != 40) & (img != 37)).astype(np.uint8) * 255
        mask.repeat(3, axis=-1)
        out_name = os.path.join(sky_dir, save_name)
        cv2.imwrite(out_name, mask)