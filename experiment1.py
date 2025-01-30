"""
experiment1.py
This script is used to generate masks using the SAM model automatic function and save them to the disk.
All generated masks are concatenated and saved in different formats for further analysis.

Usage:
- Ensure that the SAM model checkpoint is saved in the specified path: 'checkpoints/sam_vit_h_4b8939.pth'.
- Place the input images in the specified folder `IMAGES_PATH`.
- Place the ground truth masks in the specified `MASK_PATH`.
- Add paths to IMAGES_PATH and MASK_PATH.

Steps:
1. Load the SAM model.
2. Iterate over each image in the `images_path`.
3. Generate masks using the SAM model.
4. Prune masks based on size and blackish threshold.
5. Save the generated masks in different formats.
6. Calculate and save the IoU scores.

Outputs:
- Generated masks in `sam_masks/overlay`.
- Raw masks in `sam_masks/raw`.
- Numpy arrays of masks in `sam_masks/raw_npy`.
- IoU scores in `sam_masks/iou_scores_micro.csv`.

Dependencies:
- copy
- os
- pathlib
- numpy
- pandas
- torch
- PIL
- sklearn
- time
- supervision
- models (custom module)
- system (custom module)

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from PIL import Image
from sklearn.metrics import jaccard_score
from time import time

import supervision as sv

from models import load_segment_anything
from system import prune_masks

IMAGES_PATH = Path() # Change this to the path of the input images
MASK_PATH = Path() # Change this to the path of the ground truth masks

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask_generator = load_segment_anything('vit_h', 'checkpoints/sam_vit_h_4b8939.pth', device=device)

    Path('sam_masks').mkdir(exist_ok=True)
    Path('sam_masks', 'overlay').mkdir(exist_ok=True)
    Path('sam_masks', 'raw').mkdir(exist_ok=True)
    Path('sam_masks', 'raw_npy').mkdir(exist_ok=True)

    images_path = IMAGES_PATH
    mask_path = MASK_PATH

    iou_scores_binary = []
    iou_scores_micro = []
    for image_name in os.listdir(images_path):
        start_time = time()
        image = Image.open(images_path / image_name)
        image_np = np.array(image)
        truth_mask = Image.open(mask_path / image_name)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5

        masks = mask_generator.generate(image_np)

        seg_masks = np.stack([mask['segmentation'] for mask in masks], axis=0)
        np.save(Path('sam_masks', 'raw_npy', Path(image_name).with_suffix('.npy')), seg_masks)

        image_area = image_np.shape[0] * image_np.shape[1]
        min_percentage, max_percentage = (1, 70)  # Minimum and maximum mask size (% of the image area)
        min_size = min_percentage / 100 * image_area  # Minimum acceptable mask size (% of the image area)
        max_size = max_percentage / 100 * image_area  # Maximum acceptable mask size (% of the image area)

        # Prune masks that are too big or too small
        masks = [mask for mask in masks if min_size < mask['area'] < max_size]

        masks = prune_masks(masks, image_np, threshold=30, blackish_threshold=0.5)

        sam_detections = sv.Detections.from_sam(masks)
        generated_sam = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(image_np, sam_detections)

        Image.fromarray(generated_sam).save(Path('sam_masks', 'overlay', image_name))

        masks = [mask['segmentation'] for mask in masks]
        if masks:
            seg_masks = np.stack(masks, axis=0)
            sam_mask = np.sum(seg_masks, axis=0, dtype=bool)
            Image.fromarray(sam_mask).save(Path('sam_masks', 'raw', image_name))
            iou_micro = jaccard_score(truth_mask, sam_mask, average='micro')
        else:
            Image.fromarray(np.zeros(image_np.shape)).save(Path('sam_masks', 'raw', image_name))
            iou_micro = 0.0
        iou_scores_micro.append(iou_micro)
        print(f'IoU Micro: {iou_micro}')

    df_iou_scores_micro = pd.DataFrame(iou_scores_micro)
    df_iou_scores_micro.index = os.listdir(images_path)
    df_iou_scores_micro.to_csv(Path('sam_masks', 'iou_scores_micro.csv'), index=False)

    print(f'Average IoU Micro: {np.mean(iou_scores_micro)}')
