import server.init
server.init.init()

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import uuid
import os
import logging

from server.taskqueue import TaskQueue

app = Flask(__name__)
CORS(app)
job_queue = TaskQueue("redis", "6379", "segmentation_queue")
logger = logging.getLogger(__name__)

@app.route("/healthcheck")
def hello_world():
    logger.info("Healthcheck run.")
    return "Hello world.", 200

@app.route("/upload", methods=["POST"])
def predict():
    if 'file' not in request.files:
        logger.info("Uploaded with no file.")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    try:
        image = Image.open(file.stream)
        image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)

        img_and_job_id = str(uuid.uuid4())
        imgname = img_and_job_id + '.jpg'
        image.save(os.path.join(os.environ['IMAGES_DIR'], imgname))
        logger.info(f"Saved image: {imgname}.")
        job_id = job_queue.add_task("inference.infer", imgname, img_and_job_id)
        return job_id, 200
    except Exception as e:
        logger.error(f"Error in parsing and submitting image: {e}.")
        return jsonify({'error': str(e)}), 400

@app.route("/status/<job_id>")
def status(job_id):
    try:
        job_status, code = job_queue.get_job_status(job_id)
    except Exception as e:
        logger.warn(f"Unable to find job {job_id}: {str(e)}.")
        return jsonify({'error': str(e)}), 404
    if job_status['done']:
        logger.info(f"Job {job_id} queried with finished status.")
    return jsonify(job_status), code
