from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import uuid
import os
import logging

import server.classification as classification
import server.segmentation as segmentation

def rel_path(pathname):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, pathname)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/healthcheck")
def hello_world():
    logger.info("Healthcheck.")
    return "", 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        logger.info("Predict with no image.")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    try:
        ip = request.environ['REMOTE_ADDR']
        logger.info(f"Received image for {ip}.")
        image = Image.open(file.stream)
        try:
            imgid = str(uuid.uuid1())
            image.save(rel_path(os.path.join("../images", imgid + ".jpg")))
            logger.info(f"Saved img as {imgid} - {ip}.")
        except Exception as e:
            logger.warn(f"Failed to save img: {imgid} - {str(e)} - {ip}")
            print(e)
        npimg = np.array(image)
        
        logger.info(f"Starting segmentation for {imgid}.")
        subimgs, rles = segmentation.predict(npimg, 0.95)
        logger.info(f"Finished segmentation: predicted {len(rles)} pitchers for {imgid}.")

        logger.info(f"Starting classification for {imgid}.")
        for i, img in enumerate(subimgs):
            rles[i]['classification_preds'] = classification.predict(img)
            logger.info(f"Predicted: {rles[i]['classification_preds']} for {imgid}.")
        logger.info(f"Finished classification for {imgid}.")
        
        return jsonify(rles), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
