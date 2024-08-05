import logging
import os

def init():
    logging.basicConfig(
        filename=os.path.join(os.environ["LOGS_DIR"], "inference.log"),
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )