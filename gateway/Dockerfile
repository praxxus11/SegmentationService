FROM registry.gitlab.com/segmentation964330/service/torchbase

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