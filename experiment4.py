'''
experiment4.py
This script applies the LangSAM model for polyp segmentation in colon images. It processes images by generating segmentation masks, bounding boxes, and detected phrases based on a descriptive text prompt, then filters and evaluates the masks against ground truth.

Usage:
- Ensure that the paths for input images and ground truth masks are correctly set in `IMAGES_PATH` and `MASK_PATH`.
- Confirm that the LangSAM class is imported properly.
- Adjust the `text_prompt` if necessary to suit the desired detection scenario.
- Run the script to generate annotated overlay images and evaluation metrics.

Steps:
1. Define a detailed text prompt for polyp detection.
2. Provide helper functions for downloading images, saving masks, and visualizing results.
3. Load images and their corresponding ground truth masks.
4. Predict segmentation masks, bounding boxes, phrases, and confidence logits using the LangSAM model.
5. Convert predicted masks to numpy arrays and filter them based on size thresholds.
6. Annotate the original image with the filtered mask overlay.
7. Calculate evaluation metrics by comparing the generated mask with the ground truth.
8. Save overlay images, metrics, and aggregated statistics to disk.

Outputs:
- Annotated overlay images in `polyp_sam/overlay`.
- CSV file with evaluation metrics: `polyp_sam/overlay_metrics.csv`.
- CSV file with aggregated statistics: `polyp_sam/overlay_statistics.csv`.

Dependencies:
- Standard libraries: pathlib, os, time, io, requests
- External libraries: pandas, torch, numpy, matplotlib, pillow, tqdm, supervision, LangSAM (segmentation model)
- Custom modules: utils (providing calculate_all_metrics and calculate_statistics)
'''

from pathlib import Path

import pandas as pd
import torch
from time import time

import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
from lang_sam import LangSAM
from tqdm import tqdm

from utils import calculate_all_metrics, calculate_statistics

import supervision as sv

IMAGES_PATH = Path() # Change this to the path of the input images dir
MASK_PATH = Path() # Change this to the path of the ground truth masks dir

# (import your actual LangSAM class if it's located in a different module)

# The path to the folder containing images

text_prompt = """A polyp is an anomalous oval shaped small bump like structure, relatively small growth or mass that develops on the inner lining of the colon or other organs.
Multiple polyps may exist in one image
"""

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Path('polyp_sam').mkdir(exist_ok=True)
    Path('polyp_sam', 'overlay').mkdir(exist_ok=True)

    images_path = IMAGES_PATH
    mask_path = MASK_PATH

    model = LangSAM()

    metrics = {}
    for image_name in tqdm(os.listdir(images_path), total=len(os.listdir(images_path))):
        start_time = time()
        image = Image.open(images_path / image_name).convert('RGB')
        image_np = np.array(image)
        truth_mask = Image.open(mask_path / image_name)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5

        masks, boxes, phrases, logits = model.predict(image, text_prompt)

        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

        image_area = image_np.shape[0] * image_np.shape[1]
        min_percentage, max_percentage = (1, 70)  # Minimum and maximum mask size (% of the image area)
        min_size = min_percentage / 100 * image_area  # Minimum acceptable mask size (% of the image area)
        max_size = max_percentage / 100 * image_area  # Maximum acceptable mask size (% of the image area)
        masks = [mask for mask in masks_np if min_size < np.sum(mask) < max_size]

        if len(masks) > 0:
            seg_masks = np.stack(masks, axis=0)
            mask = np.sum(seg_masks, axis=0, dtype=bool)
            detections = sv.Detections(np.array([[0, 0, 0, 0]]), np.array([mask]), class_id=np.array([1]))
            mask_image = sv.MaskAnnotator().annotate(image_np.copy(), detections)
            Image.fromarray(mask_image).save(Path('polyp_sam', 'overlay', image_name))
            metrics[image_name] = calculate_all_metrics(truth_mask.ravel(), mask.ravel())
        else:
            metrics[image_name] = calculate_all_metrics(truth_mask.ravel(), np.zeros_like(truth_mask, dtype=bool).ravel())

    statistics = calculate_statistics(metrics)

    pd.DataFrame(metrics).to_csv(Path('polyp_sam') / 'overlay_metrics.csv')
    statistics.to_csv(Path('polyp_sam') / 'overlay_statistics.csv')
