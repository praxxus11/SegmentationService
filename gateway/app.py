# Setting up logging configs.
import logging
import os

def rel_path(pathname):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, pathname)

logging.basicConfig(
    filename=rel_path("./logs/app.log"),
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Actual server stuff.
from server.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
else:
    gunicorn_app = app