import logging
import os
import torch

def init():
    logging.basicConfig(
        filename=os.path.join(os.environ["LOGS_DIR"], "inference.log"),
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    torch.set_num_threads(4)