FROM python:3.11.9-slim-bullseye

WORKDIR /home

RUN apt-get update

# Unfortunately need opencv for pointrend.
RUN pip install opencv-python-headless
RUN apt-get install --assume-yes g++
RUN apt-get install --assume-yes git
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2
