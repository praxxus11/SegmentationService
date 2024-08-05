FROM python:3.12.4-alpine3.19

WORKDIR /home

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV WORKING_DIR="/home"
ENV LOGS_DIR="/home/artifacts/logs/"
ENV IMAGES_DIR="/home/artifacts/images/"

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:gunicorn_app"]