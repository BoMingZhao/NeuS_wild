import ipdb
from models.colmap_utils import read_images_binary, read_cameras_binary

imdata = read_images_binary('/data1/zhaoboming/Cambridge/KingsCollege/model_train/images.bin')
camdata = read_cameras_binary('/data1/zhaoboming/Cambridge/KingsCollege/model_train/cameras.bin')
img_path_to_id = {}
img_ids = []
image_paths = {} # {id: filename}
for v in imdata.values():
    img_path_to_id[v.name] = v.id
    image_paths[v.id] = v.name
    img_ids += [v.id]

ipdb.set_trace()