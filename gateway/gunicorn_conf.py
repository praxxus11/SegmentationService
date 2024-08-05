import os

bind = 'gateway:8000'
accesslog = os.path.join(os.environ["LOGS_DIR"], 'gunicorn.log')
errorlog = os.path.join(os.environ["LOGS_DIR"], 'gunicorn.log')
workers = 1