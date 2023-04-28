import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_predictor():
    # Inference with a panoptic segmentation model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def test_img():
    cfg, predictor = get_predictor()
    # im = cv2.imread("./input.jpg")
    im = cv2.imread("/mnt/nas_8/group/BBYang/nerf_w_boming/sky_seg/trevi_fountain/00644051_242819650.jpg")
    # cv2.imshow('i', im)
    # cv2.waitKey(0)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2.imshow('aa', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    from tqdm import tqdm
    import glob
    import ipdb
    scene_list = ['GreatCourt', 'StMarysChurch', 'ShopFacade', 'KingsCollege']
    for scene in scene_list:
        input_path = f"/data1/zhaoboming/Cambridge/{scene}/images"
        output_dir = f'/data1/zhaoboming/Cambridge/{scene}/segment'
        os.makedirs(output_dir, exist_ok=True)
        imgs_path = []
        for seq_name in os.listdir(input_path):
            os.makedirs(os.path.join(output_dir, seq_name), exist_ok=True)
            imgs_path.extend(glob.glob(os.path.join(input_path, seq_name, "*.png")))
        cfg, predictor = get_predictor()
        for img_path in tqdm(imgs_path):
            img_name = img_path.split('/')[-1]
            seq_name = img_path.split('/')[-2]
            img_prefix = seq_name + '/' + img_name[:-4]
            im = cv2.imread(img_path)
            panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            cv2.imwrite(os.path.join(output_dir, img_prefix+'_colorseg.png'), out.get_image()[:, :, ::-1])

            # save raw segmentation category label
            # import ipdb; ipdb.set_trace()
            seg_category = torch.zeros_like(panoptic_seg, dtype=torch.int16)

            for seg_ in segments_info:
                id_ = seg_['id']
                category_id = seg_['category_id']
                seg_category[panoptic_seg == id_] = category_id

            cv2.imwrite(os.path.join(output_dir, img_prefix+'_category.png'), seg_category.detach().cpu().numpy().astype(np.uint16))
