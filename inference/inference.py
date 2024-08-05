import os
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import logging

from segmentation.inference import predict as predict_segmentation
from classification.inference import predict as predict_classification

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

    logger.info(f"Starting segmentation for {jpg_filename}.")
    numpy_binarymasks = predict_segmentation(numpy_image, 0.95)
    logger.info(f"Finished segmentation for {jpg_filename} - Predicted {len(numpy_binarymasks)} pitchers.")

    output = []
    for i, binary_mask in enumerate(numpy_binarymasks):
        current_pitcher_pred = {}

        logger.info(f"Starting classification for {jpg_filename} - mask #{i+1}.")
        classification_results = predict_classification(numpy_image, binary_mask['numpy_mask'], 5)
        current_pitcher_pred["classification"] = predict_classification(numpy_image, binary_mask['numpy_mask'], 5)
        logger.info(f"Finished classification for {jpg_filename} - mask #{i+1} - {classification_results}.")

        current_pitcher_pred["segmentation"] = {
            'mask': binary_numpy_arr_to_base64png(binary_mask['numpy_mask']),
            'confidence': binary_mask['confidence']
        }
        output.append(current_pitcher_pred)
    return output