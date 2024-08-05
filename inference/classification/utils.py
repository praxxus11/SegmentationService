from PIL import Image
import numpy as np

def cut_out_mask(numpy_img, numpy_mask):
    temp_img = np.copy(numpy_img)
    temp_img[~numpy_mask] = 0
    p = np.where(temp_img != 0)
    temp_img = temp_img[min(p[0]) : max(p[0]) + 1, min(p[1]) : max(p[1]) + 1]
    return temp_img

def get_topk_predictions(raw_output_list, id_to_species_name, topk):
    labeled_preds = []
    for i in range(len(raw_output_list)):
        labeled_preds.append({
            'species': id_to_species_name[i],
            'confidence': raw_output_list[i]
        })
    labeled_preds.sort(key=lambda x: x['confidence'], reverse=True)
    return labeled_preds[:topk]