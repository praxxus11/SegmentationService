from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import os
import logging

logger = logging.getLogger(__name__)

def build_segmentation_model():
    logger.info("Started building segmentation model.")
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    path_to_model_cfg = "detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
    cfg.merge_from_file(os.path.join(os.environ["WORKING_DIR"], path_to_model_cfg))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cpu'
    
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    model_path = os.path.join(os.environ["MODELS_DIR"], os.environ["SEGMENTATION_MODEL_NAME"])
    checkpointer.load(model_path)
    model.cpu()
    model.eval()
    logger.info("Done building segmentation model.")
    return model

segmentation_model = build_segmentation_model()
