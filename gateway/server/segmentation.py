from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import torch
import torchvision.transforms.functional as F
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def rel_path(pathname):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, pathname)

def get_model(path=None):
    logger.info("Loading segmentation model.")
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(rel_path("../detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cpu'
    
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(path)
    logger.info("Done loading segmentation model.")
    return model

model = get_model(rel_path('./models/pointrend_weights.pth'))

# Given 0/1 numpy array masks and img, cuts out the actual imgs
def get_numpy_imgs_from_masks(np_img, masks):
    numpyimgs = []
    for i in range(len(masks)):
        temp_img = np.copy(np_img)
        mask_matrix, confidence = masks[i]
        temp_img[~mask_matrix] = 0
        p = np.where(temp_img != 0)
        temp_img = temp_img[min(p[0]) : max(p[0]) + 1, min(p[1]) : max(p[1]) + 1]
        numpyimgs.append(temp_img)
    return numpyimgs

def toRLE(tensor):
    flattened = tensor.flatten().int()
    init = 't' if flattened[0]==1 else 'f'
    prependvalue = [0] if flattened[0]==1 else [1]
    diffs = flattened.diff(prepend=torch.tensor(prependvalue))
    change_idxs = torch.logical_or(diffs == 1, diffs == -1).nonzero().flatten()
    change_idxs = torch.cat((change_idxs, torch.tensor([len(diffs)])))
    rle = init + ','.join(map(str, change_idxs.diff().tolist()))
    return {'rle': rle, 'height': tensor.shape[0], 'width': tensor.shape[1]}

def getRLES(mask_list):
    res = []
    for mask, _ in mask_list:
        res.append(toRLE(mask))
    return res

@torch.no_grad()
def make_masks_prediction(img_as_np, threshold):
    global model
    # img = Image.open(imgname).convert('RGB')
    # width, height = img.size
    height, width = img_as_np.shape[:2]
    tens_img = torch.as_tensor(img_as_np.astype("float32").transpose(2, 0, 1)).cpu()
    tens_img = F.resize(tens_img, 700, antialias=True)

    model.cpu()
    model.eval()

    outputs = model([{"image": tens_img, "height": height, "width": width }])
    outputs = outputs[0]['instances']
    
    mask_list = []
    masks = outputs.pred_masks 
    for i in range(len(masks)):
        if outputs.scores[i] < threshold:
            continue
        mask_list.append((outputs.pred_masks[i].cpu().squeeze(), outputs.scores[i].cpu()))
    
    rles = getRLES(mask_list)
    return get_numpy_imgs_from_masks(img_as_np, mask_list), rles

def predict(npimg, threshold):
    return make_masks_prediction(npimg, threshold)
