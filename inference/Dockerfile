FROM python:3.11.9-slim-bullseye

WORKDIR /home

RUN apt-get update

# Unfortunately need opencv for pointrend.
RUN pip install opencv-python-headless --no-cache-dir
RUN apt-get install --assume-yes g++
RUN apt-get install --assume-yes git
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2 --no-cache-dir

#######################   Done installing deps   ###################

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV WORKING_DIR="/home"
ENV LOGS_DIR="/home/artifacts/logs/"
ENV IMAGES_DIR="/home/artifacts/images/"
ENV MODELS_DIR="/home/artifacts/models/"
ENV DB_DIR="/home/artifacts/db/"
ENV SEGMENTATION_MODEL_NAME="pointrend_weights.pth"
ENV CLASSIFICATION_MODEL_NAME="swinv2_weights.pth"
ENV SPECIES_LIST="species.json"

CMD ["python", "worker.py"]