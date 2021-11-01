from .Flicker_8k import Flicker8k
from .COCO import COCO


def dataset(image_path, text_path, cap_file, img_name, dataset, cfg):
    if dataset == 'COCO':
        f_data = COCO(image_path, text_path, cap_file, img_name, cfg)
    else:
        f_data = Flicker8k(image_path, text_path, cap_file, img_name, cfg)

    return f_data
