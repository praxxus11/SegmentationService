import torch
import torchvision
import torchvision.transforms.v2 as T
import json
import os
import logging

logger = logging.getLogger(__name__)

def build_classification_model(num_output_classes):
    logger.info("Started building classification model.")
    model = torchvision.models.swin_v2_t()
    inp_feats = model.head.in_features
    model.head = torch.nn.Sequential(
        torch.nn.Linear(inp_feats, num_output_classes),
    )
    model_path = os.path.join(os.environ["MODELS_DIR"], os.environ["CLASSIFICATION_MODEL_NAME"])
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
    model.cpu()
    model.eval()
    logger.info("Done building classification model.")
    return model

def get_classification_transforms():
    transforms_list = []
    transforms_list.append(T.ToImage())
    transforms_list.append(T.ToDtype(torch.uint8, scale=True))
    transforms_list.append(T.Resize((256,256)))        
    transforms_list.append(T.ToDtype(torch.float32, scale=True))
    transforms_list.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms_list)

def get_classes_info():
    with open(os.path.join(os.environ["MODELS_DIR"], os.environ["SPECIES_LIST"])) as f:
        species_list = json.load(f)
    species_name_to_id = {}
    id_to_species_name = {}
    for i, species in enumerate(species_list):
        species_name_to_id[species] = i
        id_to_species_name[i] = species
    return species_name_to_id, id_to_species_name, len(species_list)

species_name_to_id, id_to_species_name, num_classes = get_classes_info()
classification_model = build_classification_model(num_classes)
classification_transforms = get_classification_transforms()