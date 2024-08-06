import os
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import time
import logging

from segmentation.inference import predict as predict_segmentation
from classification.inference import predict as predict_classification
from storage.meta import Meta, ClassificationMeta
from storage.db import dump_meta

logger = logging.getLogger(__name__)

def binary_numpy_arr_to_base64png(binary_numpy_array):
    assert len(binary_numpy_array.shape) == 2

    pil_img = Image.fromarray(binary_numpy_array)
    img_io = BytesIO()
    pil_img.save(img_io, 'png', optimize=True)
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64

def infer(jpg_filename):
    jpg_image = Image.open(os.path.join(os.environ["IMAGES_DIR"], jpg_filename))
    numpy_image = np.array(jpg_image)

    seg_meta = Meta()
    seg_meta.img_id = jpg_filename.split(".")[0]

    logger.info(f"Starting segmentation for {jpg_filename}.")
    seg_meta.start_mili = time.time_ns() // 1_000_000
    numpy_binarymasks = predict_segmentation(numpy_image, 0.95)
    seg_meta.end_mili = time.time_ns() // 1_000_000
    logger.info(f"Finished segmentation for {jpg_filename}.")

    seg_meta.num_masks = len(numpy_binarymasks)

    logger.info(f"Starting classification for {jpg_filename}.")
    output = []
    for i, binary_mask in enumerate(numpy_binarymasks):
        clas_meta = ClassificationMeta()
        clas_meta.pitcher_id = str(i)
        current_pitcher_pred = {}

        clas_meta.start_mili = time.time_ns() // 1_000_000
        classification_results = predict_classification(numpy_image, binary_mask['numpy_mask'], 5)
        clas_meta.end_mili = time.time_ns() // 1_000_000

        clas_meta.pred_species_1 = classification_results[0]['species']
        clas_meta.pred_species_1_conf = classification_results[0]['confidence']
        clas_meta.pred_species_2 = classification_results[1]['species']
        clas_meta.pred_species_2_conf = classification_results[1]['confidence']
        clas_meta.pred_species_3 = classification_results[2]['species']
        clas_meta.pred_species_3_conf = classification_results[2]['confidence']

        current_pitcher_pred["classification"] = classification_results
        current_pitcher_pred["segmentation"] = {
            'mask': binary_numpy_arr_to_base64png(binary_mask['numpy_mask']),
            'confidence': binary_mask['confidence']
        }
        output.append(current_pitcher_pred)
        seg_meta.classifications.append(clas_meta)
    logger.info(f"Finished classification for {jpg_filename}.")
    dump_meta(seg_meta)
    return output