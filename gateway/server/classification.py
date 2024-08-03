import torch
import torchvision
import torchvision.transforms.v2 as T
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

with open(os.path.join(os.environ["MODELS_DIR"], os.environ["SPECIES_LIST"])) as f:
    SPECIES_LIST = json.load(f)
NUM_CLASSES = len(SPECIES_LIST)

def get_classification_model():
    global NUM_CLASSES
    global DEVICE
    logger.info("Loading classification model.")
    model = torchvision.models.swin_v2_t()
    inp_feats = model.head.in_features
    model.head = torch.nn.Sequential(
        torch.nn.Linear(inp_feats, NUM_CLASSES),
    )
    model = model.cpu()
    model_path = os.path.join(os.environ["MODELS_DIR"], os.environ["CLASSIFICATION_MODEL_NAME"])
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
    logger.info("Done loading classification model.")
    return model

def get_transforms():
    transforms_list = []
    transforms_list.append(T.ToImage())
    transforms_list.append(T.ToDtype(torch.uint8, scale=True))
    transforms_list.append(T.Resize((256,256)))        
    transforms_list.append(T.ToDtype(torch.float32, scale=True))
    transforms_list.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms_list)

@torch.no_grad()
def get_prediction(model, transforms, numpy_img):
    model.eval()
    model.cpu()
    transformed_img_tensor = transforms(numpy_img)
    transformed_img_tensor = transformed_img_tensor.cpu()
    transformed_img_tensor = transformed_img_tensor.unsqueeze(0)

    logger.info("Before classification model.")
    output = model(transformed_img_tensor)[0]
    logger.info("Right after classification model.")

    output = torch.nn.functional.softmax(output, dim=0)
    return output

def get_species_name_maps(sp_list):
    name_to_label = {}
    label_to_name = {}
    for i, species in enumerate(sp_list):
        name_to_label[species] = i
        label_to_name[i] = species
    return name_to_label, label_to_name

def match_label_to_raw_output(output, topn):
    global label_to_name
    preds = []
    for i in range(len(output)):
        preds.append((label_to_name[i], output[i].item()))
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:topn]

name_to_label, label_to_name = get_species_name_maps(SPECIES_LIST)
transforms = get_transforms()
model = get_classification_model()

def predict(npimg):
    global model
    raw_outp = get_prediction(model, transforms, npimg)
    cleaned = match_label_to_raw_output(raw_outp, 5)
    return cleaned
