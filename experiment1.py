"""
experiment1.py
This script uses the SAM model to automatically generate segmentation masks from input images and evaluates them by computing the Intersection over Union (IoU) against ground truth masks. The raw masks are saved as numpy arrays, and annotated overlay images are generated for visual inspection.

Usage:
- Update the paths for input images and ground truth masks in `IMAGES_PATH` and `MASK_PATH`.
- Ensure that the SAM model checkpoint is available at 'checkpoints/sam_vit_h_4b8939.pth'.
- Run the script to process the images, generate and prune masks, annotate the original images, and calculate IoU metrics.

Steps:
1. Load the SAM model using a custom loader function.
2. Create necessary output directories: `sam_masks/overlay`, `sam_masks/raw`, and `sam_masks/raw_npy`.
3. For each image:
   - Load the image and its corresponding ground truth mask.
   - Generate segmentation masks using the SAM model.
   - Save raw, unfiltered masks as numpy arrays.
   - Prune the generated masks based on area and a blackish threshold.
   - Annotate the image with the pruned masks.
   - Merge the masks and compute the micro IoU using the Jaccard score.
4. Save the IoU scores to a CSV file.

Outputs:
- Overlay images with annotated masks in `sam_masks/overlay`.
- Raw binary mask images in `sam_masks/raw`.
- Numpy arrays of raw masks in `sam_masks/raw_npy`.
- CSV file with IoU scores: `sam_masks/iou_scores_micro.csv`.

Dependencies:
- Standard libraries: os, pathlib, time
- External libraries: numpy, pandas, torch, PIL, sklearn, supervision
- Custom modules: models (load_segment_anything) and utils (prune_masks_if_blackish, prune_masks_outside_area)
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
from utils import prune_masks_if_blackish, prune_masks_outside_area

IMAGES_PATH = Path() # Change this to the path of the input images dir
MASK_PATH = Path() # Change this to the path of the ground truth masks dir
segment_anything_model_checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask_generator = load_segment_anything('vit_h', segment_anything_model_checkpoint, device=device)

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

        # Load images and ground truth masks
        image = Image.open(images_path / image_name)
        image_np = np.array(image)
        truth_mask = Image.open(mask_path / image_name)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5

        # Generate automatically masks
        masks = mask_generator.generate(image_np)

        # Save raw, unfiltered masks as numpy arrays for later use
        seg_masks = np.stack([mask['segmentation'] for mask in masks], axis=0)
        np.save(Path('sam_masks', 'raw_npy', Path(image_name).with_suffix('.npy')), seg_masks)

        # Prune masks
        masks = prune_masks_outside_area(image_np.shape[0] * image_np.shape[1], masks)
        masks = prune_masks_if_blackish(masks, image_np, threshold=30, blackish_threshold=0.5)

        # Annotate masks
        sam_detections = sv.Detections.from_sam(masks)
        generated_sam = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(image_np, sam_detections)

        Image.fromarray(generated_sam).save(Path('sam_masks', 'overlay', image_name))

        # Merge masks and calculate IoU
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
