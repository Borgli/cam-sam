'''
utils.py
This module provides utility functions for image and mask processing, evaluation metric calculations, and experiment management for segmentation tasks. It includes functions to generate unique experiment names, load and preprocess images and masks, compute bounding boxes, store and retrieve GradCAM and SAM outputs, prune masks based on quality criteria, and calculate both detailed and aggregate evaluation statistics.

Usage:
- Import the module to access helper functions for:
  • Generating experiment names with current timestamps.
  • Loading images (with resizing and normalization) and masks.
  • Computing bounding boxes from binary masks.
  • Determining file paths for stored GradCAM and SAM outputs.
  • Saving and retrieving GradCAM data (in .npy format) and SAM results (in .pkl format).
  • Calculating segmentation metrics including IoU, Dice score, precision, recall, MCC, and pixel accuracy.
  • Aggregating metrics into statistical summaries (mean, median, standard deviation, etc.).
  • Pruning masks based on the proportion of “blackish” pixels or area constraints.
  • Creating experiment folders with timestamped names.
  • Finding the best overlapping mask using IoU.
- It also defines a custom callback class (ProgressCounterCallback) to track optimization steps.

Steps:
1. Use `generate_experiment_name` to create a unique experiment identifier.
2. Load and preprocess images via `load_image` and masks via `load_mask`.
3. Compute bounding boxes on masks using `find_bounding_box`.
4. Manage GradCAM and SAM outputs with `find_grad_cam_path`, `get_stored_gradcam`, `store_gradcam`, `get_stored_sam`, and `store_sam`.
5. Calculate per-image metrics with `calculate_all_metrics`.
6. Aggregate metrics with `calculate_statistics` for further analysis.
7. Optionally prune masks using `prune_masks_if_blackish` and `prune_masks_outside_area`.
8. Create new experiment folders using `create_experiment_folder`.
9. Identify the best overlapping mask by calling `find_overlap_with_iou`.

Dependencies:
- Standard libraries: os, pickle, datetime, pathlib
- External libraries: pandas, torchvision, cv2, numpy, supervision, scikit-metrics
'''

import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from torchvision import transforms

import cv2
import numpy as np
import supervision as sv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef


def generate_experiment_name(model_type, dataset_name, additional_info):
    # Get current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    # Combine date and time with model type, dataset name and additional info
    experiment_name = f"{date_time}_{model_type}_{dataset_name}_{additional_info}"

    return experiment_name


def load_image(image_path, size, device):
    images = image_path if type(image_path) == list else [image_path]
    image_list = []
    for image in images:
        rgb_img = cv2.imread(image, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, size)
        raw_image = rgb_img.copy()
        rgb_img = np.float32(rgb_img) / 255
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4762, 0.3054, 0.2368],
                                 [0.3345, 0.2407, 0.2164])
        ])
        image_list.append((cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), normalize(rgb_img).unsqueeze(0).to(device)))

    return image_list if type(image_path) == list else image_list[0]


def load_mask(mask_path, size):
    rgb_mask = cv2.imread(mask_path, 1)[:, :, ::-1]
    rgb_mask = cv2.resize(rgb_mask, size)
    gray_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    gray_mask_reshaped = np.reshape(gray_mask, (1, size[0], size[1]))
    gray_mask_reshaped = np.float32(gray_mask_reshaped) / 255
    rgb_mask = np.float32(rgb_mask) / 255
    return gray_mask_reshaped, rgb_mask


def find_bounding_box(mask, threshold):
    filtered_mask = mask > threshold
    reshaped_filtered_mask = np.expand_dims(filtered_mask, axis=0)

    # Create input_box in xyxy format from grayscale_cam
    return sv.detection.utils.mask_to_xyxy(reshaped_filtered_mask)


def find_grad_cam_path(image_path, dataset_roots, generated_gradcam_dir):
    original_path = Path(image_path)

    dataset_root = None
    for part in original_path.parts:
        if part in dataset_roots:
            dataset_root = original_path.parents[original_path.parts.index(part)]
            break

    if dataset_root is None:
        raise ValueError("Dataset root not found in the provided path.")

    return generated_gradcam_dir / original_path.relative_to(dataset_root)


def get_stored_gradcam(base_path, image_path):
    grad_cam_path = base_path / image_path.name
    grad_cam_path = grad_cam_path.with_suffix('.npy')
    if not grad_cam_path.exists():
        return None
    else:
        return np.load(grad_cam_path.with_suffix('.npy'))


def store_gradcam(image_path, cam_descriptor, gradcam):
    grad_cam_path = find_grad_cam_path(image_path)
    grad_cam_path = grad_cam_path.parent / cam_descriptor / grad_cam_path.name
    grad_cam_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grad_cam_path.with_suffix('.npy'), gradcam)


def get_stored_sam(image_path):
    sam_path = find_grad_cam_path(image_path)
    sam_path = sam_path.parent / 'sam_automatic_generation' / sam_path.name
    sam_path = sam_path.with_suffix('.pkl')
    if not sam_path.exists():
        return None
    else:
        with open(sam_path.with_suffix('.pkl'), 'rb') as f:
            return pickle.load(f)

def store_sam(image_path, sam):
    sam_path = find_grad_cam_path(image_path)
    sam_path = sam_path.parent / 'sam_automatic_generation' / sam_path.name
    sam_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sam_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(sam, f)


def calculate_all_metrics(gt_mask, mask):
    iou = jaccard_score(gt_mask, mask, average='binary')
    dice = f1_score(gt_mask, mask, average='binary')
    precision = precision_score(gt_mask, mask, average='binary', zero_division=0)
    recall = recall_score(gt_mask, mask, average='binary')
    mcc = matthews_corrcoef(gt_mask, mask)
    pixel_accuracy = accuracy_score(gt_mask, mask)

    #tn, fp, fn, tp = confusion_matrix(gt_mask, mask).ravel()
    #specificity = tn / (tn + fp)
    #sensitivity = recall_score(gt_mask, mask, average='binary')
    #balanced_accuracy = (sensitivity + specificity) / 2
    #f2_score = (5 * precision_score(gt_mask, mask, average='binary') * sensitivity) / (
    #        4 * precision_score(gt_mask, mask, average='binary') + sensitivity)

    # Assuming y_true and y_pred are binary masks with values 0 and 1
    #y_true_points = np.argwhere(gt_mask == 1)
    #y_pred_points = np.argwhere(mask == 1)

    #hausdorff_distance = max(directed_hausdorff(y_true_points, y_pred_points)[0],
    #                         directed_hausdorff(y_pred_points, y_true_points)[0])

    #ssim_value = ssim(gt_mask, mask)

    #fpr = fp / (fp + tn)
    #fnr = fn / (fn + tp)
    #ber = (fnr + fpr) / 2

    image_metrics = {
        'IoU / Jaccard\'s index': iou,
        'Dice / F1 Score': dice,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc,
        'Pixel Accuracy': pixel_accuracy#,
        #'Specificity': specificity,
        #'Sensitivity': sensitivity,
        #'Balanced_accuracy': balanced_accuracy,
        #'F2 score': f2_score,
        #'Hausdorff Distance': hausdorff_distance,
        #'SSIM': ssim_value,
        #'False Positive Rate': fpr,
        #'False Negative Rate': fnr,
        #'Balanced Error Rate': ber
    }

    return image_metrics


def calculate_statistics(step_metrics):
    step_metrics = pd.DataFrame(step_metrics).T
    mean_metrics = step_metrics.mean().to_frame(name='Mean').T
    median_metrics = step_metrics.median().to_frame(name='Median').T
    std_metrics = step_metrics.std().to_frame(name='Standard Deviation').T
    variance_metrics = step_metrics.var().to_frame(name='Variance').T
    min_metrics = step_metrics.min().to_frame(name='Minimum').T
    max_metrics = step_metrics.max().to_frame(name='Maximum').T
    range_metrics = (step_metrics.max() - step_metrics.min()).to_frame(name='Range').T
    q1_metrics = step_metrics.quantile(0.25).to_frame(name='25th Percentile').T
    q3_metrics = step_metrics.quantile(0.75).to_frame(name='75th Percentile').T
    iqr_metrics = (step_metrics.quantile(0.75) - step_metrics.quantile(0.25)).to_frame(name='Interquartile Range').T
    skewness_metrics = step_metrics.skew().to_frame(name='Skewness').T
    kurtosis_metrics = step_metrics.kurtosis().to_frame(name='Kurtosis').T

    all_combined_metrics = pd.concat([
        mean_metrics, median_metrics, std_metrics, variance_metrics,
        min_metrics, max_metrics, range_metrics, q1_metrics, q3_metrics,
        iqr_metrics, skewness_metrics, kurtosis_metrics
    ])

    return all_combined_metrics


def is_blackish(pixel, threshold):
    return np.all(pixel < threshold)

def calculate_blackish_proportion(mask, image, threshold):
    height, width = mask.shape[0], mask.shape[1]
    blackish_pixels = 0
    total_mask_pixels = np.sum(mask)

    for i in range(height):
        for j in range(width):
            if mask[i, j] and is_blackish(image[i, j], threshold):
                blackish_pixels += 1

    return blackish_pixels / total_mask_pixels if total_mask_pixels > 0 else 0

def prune_masks_if_blackish(masks, image, threshold=30, blackish_threshold=0.5):
    pruned_masks = []
    for mask in masks:
        blackish_proportion = calculate_blackish_proportion(mask['segmentation'], image, threshold)
        if blackish_proportion < blackish_threshold:
            pruned_masks.append(mask)
    return pruned_masks


def prune_masks_outside_area(image_area, masks):
    min_percentage, max_percentage = (1, 70)  # Minimum and maximum mask size (% of the image area)
    min_size = min_percentage / 100 * image_area  # Minimum acceptable mask size (% of the image area)
    max_size = max_percentage / 100 * image_area  # Maximum acceptable mask size (% of the image area)

    # Prune masks that are too big or too small
    return [mask for mask in masks if min_size < mask['area'] < max_size]


def create_experiment_folder(base_path, identifier):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the folder name
    folder_name = f"{identifier}_{timestamp}"

    # Create the full path
    full_path = os.path.join(base_path, folder_name)

    # Create the directory
    os.makedirs(full_path, exist_ok=True)

    return full_path


def find_overlap_with_iou(masks, predictor_mask):
    best_mask = None
    best_iou = 0

    for mask in masks:
        # Calculate the overlap between the mask and the predictor_mask
        intersection = np.sum(mask['segmentation'] & predictor_mask)
        union = np.sum(mask['segmentation'] | predictor_mask)
        iou = intersection / union if union > 0 else 0

        # If the overlap is higher than the best_overlap, update the best_mask and best_overlap
        if iou > best_iou:
            best_mask = mask
            best_iou = iou

    return best_mask, best_iou


# Define a custom callback to track the current step
class ProgressCounterCallback:
    def __init__(self):
        self.current_step = 0

    def __call__(self, res):
        self.current_step += 1
