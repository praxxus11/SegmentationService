from classification.setup import classification_model, classification_transforms, species_name_to_id, id_to_species_name, num_classes
from classification.utils import cut_out_mask, get_topk_predictions

import torch
import torchvision
import torchvision.transforms.v2 as T
import numpy as np

@torch.no_grad()
def run_classification_model(model, transforms, numpy_img):
    model.eval()
    model.cpu()

    transformed_img_tensor = transforms(numpy_img)
    transformed_img_tensor = transformed_img_tensor.cpu()
    transformed_img_tensor = transformed_img_tensor.unsqueeze(0)

    output = model(transformed_img_tensor)[0]
    output = torch.nn.functional.softmax(output, dim=0)
    output = output.tolist()

    assert len(output) == num_classes
    return output

def predict(numpy_input_image, numpy_binary_mask, topk):
    assert numpy_input_image.shape[:2] == numpy_binary_mask.shape

    numpy_input_cutout = cut_out_mask(numpy_input_image, numpy_binary_mask)
    raw_predictions = run_classification_model(classification_model, classification_transforms, numpy_input_cutout)
    cleaned_predictions = get_topk_predictions(raw_predictions, id_to_species_name, 5)
    return cleaned_predictions
