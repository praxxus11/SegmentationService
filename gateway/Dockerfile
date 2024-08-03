FROM python:3.11.9-slim-bullseye

WORKDIR /home

RUN apt-get update

# Require cv2 for detectron2.projects.point_rend
RUN apt-get install --assume-yes python3-opencv
RUN pip install opencv-python

# Require g++ for detectron2
RUN apt-get install --assume-yes g++
RUN apt-get install --assume-yes git

RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git
# Putting these here instead of requirements for caching.
RUN pip install torch==2.3.0
RUN pip install torchvision==0.18.0
RUN python -m pip install -e detectron2

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV WORKING_DIR="/home"
ENV LOGS_DIR="/home/artifacts/logs/"
ENV IMAGES_DIR="/home/artifacts/images/"
ENV MODELS_DIR="/home/artifacts/models/"
ENV SEGMENTATION_MODEL_NAME="pointrend_weights.pth"
ENV CLASSIFICATION_MODEL_NAME="swinv2_weights.pth"
ENV SPECIES_LIST="species.json"

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:gunicorn_app"]