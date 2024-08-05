from segmentation.setup import segmentation_model

import torch
import torchvision.transforms.functional as F
import numpy as np

@torch.no_grad()
def run_segmentation_model(model, numpy_input_image, threshold):
    model.cpu()
    model.eval()

    height, width = numpy_input_image.shape[:2]
    tensor_input_image = torch.as_tensor(numpy_input_image.astype("float32").transpose(2, 0, 1)).cpu()
    tensor_input_image = F.resize(tensor_input_image, 700, antialias=True)

    outputs = model([{"image": tensor_input_image, "height": height, "width": width }])
    outputs = outputs[0]['instances']

    numpy_binary_masks = []
    for i in range(len(outputs.pred_masks)):
        if outputs.scores[i] >= threshold:
            numpy_binary_mask = np.array(outputs.pred_masks[i].cpu().squeeze())
            numpy_binary_masks.append({
                "numpy_mask": numpy_binary_mask,
                "confidence": outputs.scores[i].cpu().item()
            })
    
    return numpy_binary_masks

def predict(numpy_input_image, threshold):
    assert len(numpy_input_image.shape) == 3
    assert threshold >= 0.0 and threshold <= 1.0

    return run_segmentation_model(segmentation_model, numpy_input_image, threshold)
