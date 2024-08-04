import server.init
server.init.init()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import uuid
import os
import logging

import server.classification as classification
import server.segmentation as segmentation

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route("/secretgui")
def gui():
    return render_template("hi.html")

@app.route("/healthcheck")
def hello_world():
    logger.info("Healthcheck.")
    return "Hello world!", 200

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
        # Need to remove alpha channel.
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        try:
            imgid = str(uuid.uuid1())
            image.save(os.path.join(os.environ['IMAGES_DIR'], imgid + ".jpg"))
            logger.info(f"Saved img as {imgid} - {ip}.")
        except Exception as e:
            logger.warn(f"Failed to save img: {imgid} - {str(e)} - {ip}")
            print(e)
        npimg = np.array(image)
        
        logger.info(f"Starting segmentation for {imgid}.")
        subimgs, base64s = segmentation.predict(npimg, 0.95)
        logger.info(f"Finished segmentation: predicted {len(base64s)} pitchers for {imgid}.")

        logger.info(f"Starting classification for {imgid}.")
        for i, img in enumerate(subimgs):
            base64s[i]['classification_preds'] = classification.predict(img)
            logger.info(f"Predicted: {base64s[i]['classification_preds']} for {imgid}.")
        logger.info(f"Finished classification for {imgid}.")
        
        return jsonify(base64s), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
