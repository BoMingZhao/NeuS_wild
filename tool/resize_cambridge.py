import cv2
import os

base_path = '/data1/zhaoboming/Cambridge'
scene_name = ['StMarysChurch']

for scene in scene_name:
    seq_path = os.path.join(base_path, scene, 'images')
    for seq in os.listdir(seq_path):
        img_path = os.path.join(seq_path, seq)
        for name in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, name))
            img = cv2.resize(img, (1024, 576))
            cv2.imwrite(os.path.join(img_path, name), img)

