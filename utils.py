import pickle
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List

import pandas as pd
from torchvision import datasets, transforms

import cv2
import numpy as np
import supervision as sv
from matplotlib import pyplot as plt

from config import Config

from scipy.spatial.distance import directed_hausdorff

from skimage.metrics import structural_similarity as ssim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef, \
    confusion_matrix


def generate_experiment_name(model_type, dataset_name, additional_info):
    # Get current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    # Combine date and time with model type, dataset name and additional info
    experiment_name = f"{date_time}_{model_type}_{dataset_name}_{additional_info}"

    return experiment_name


# Function that inputs the output and plots image and mask
def show_output(result_dict, axes=None):
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
    # Plot for each segment area
    for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, mask * 0.5)))


def draw_masks(
        image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7
) -> np.ndarray:
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image


def draw_segmentation_map(
        image: np.ndarray, segmentation_map: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    colored_mask = np.expand_dims(segmentation_map, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=(255, 0, 0))
    image_overlay = masked.filled()
    return cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)


def draw_segmentation_map_with_contour(
        image: np.ndarray, segmentation_map: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    colored_mask = np.expand_dims(segmentation_map, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=(255, 0, 0))
    image_overlay = masked.filled()
    image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    contours, _ = cv2.findContours(
        np.uint8(segmentation_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image


def draw_segmentation_map_with_contour_and_labels(
        image: np.ndarray,
        segmentation_map: np.ndarray,
        labels: List[str],
        alpha: float = 0.7,
) -> np.ndarray:
    colored_mask = np.expand_dims(segmentation_map, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=(255, 0, 0))
    image_overlay = masked.filled()
    image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    contours, _ = cv2.findContours(
        np.uint8(segmentation_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(image.shape[1] * px, image.shape[0] * px))
    plt.rcParams["legend.fontsize"] = int(14 * image.shape[0] / 256 / max(1, len(labels) / 6))
    lw = 5 * image.shape[0] / 256
    lines = [plt.Line2D([0], [0], color=(255, 0, 0), lw=lw) for _ in range(len(labels))]
    plt.legend(
        lines,
        labels,
        mode="expand",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.axis("off")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, (image.shape[1], image.shape[0]))
    return np.hstack((image, data))


def draw_segmentation_map_with_contour_and_labels_and_colorbar(
        image: np.ndarray,
        segmentation_map: np.ndarray,
        labels: List[str],
        alpha: float = 0.7,
) -> np.ndarray:
    colored_mask = np.expand_dims(segmentation_map, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=(255, 0, 0))
    image_overlay = masked.filled()
    image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    contours, _ = cv2.findContours(
        np.uint8(segmentation_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(image.shape[1] * px, image.shape[0] * px))
    plt.rcParams["legend.fontsize"] = int(14 * image.shape[0] / 256 / max(1, len(labels) / 6))
    lw = 5 * image.shape[0] / 256
    lines = [plt.Line2D([0], [0], color=(255, 0, 0), lw=lw) for _ in range(len(labels))]
    plt.legend(
        lines,
        labels,
        mode="expand",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.axis("off")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, (image.shape[1], image.shape[0]))
    return np.hstack((image, data))


def draw_segmentation_map_with_contour_and_labels_and_colorbar_and_scale(
        image: np.ndarray,
        segmentation_map: np.ndarray,
        labels: List[str],
        alpha: float = 0.7,
        scale: float = 1.0,
) -> np.ndarray:
    colored_mask = np.expand_dims(segmentation_map, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=(255, 0, 0))
    image_overlay = masked.filled()
    image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    contours, _ = cv2.findContours(
        np.uint8(segmentation_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(image.shape[1] * px, image.shape[0] * px))
    plt.rcParams["legend.fontsize"] = int(14 * image.shape[0] / 256 / max(1, len(labels) / 6))
    lw = 5 * image.shape[0] / 256
    lines = [plt.Line2D([0], [0], color=(255, 0, 0), lw=lw) for _ in range(len(labels))]
    plt.legend(
        lines,
        labels,
        mode="expand",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.axis("off")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, (image.shape[1], image.shape[0]))
    result = np.hstack((image, data))

    scale = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    return np.vstack((result, scale))


def draw_mask_from_mask(mask: np.ndarray, color: np.ndarray = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])) -> np.ndarray:
    h, w = mask.shape[-2:]
    return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    dice = 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))
    return dice


def calculate_precision(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    precision = np.sum(intersection) / np.sum(mask1)
    return precision


def calculate_recall(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    recall = np.sum(intersection) / np.sum(mask2)
    return recall


def calculate_f1_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    precision = calculate_precision(mask1, mask2)
    recall = calculate_recall(mask1, mask2)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def calculate_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    accuracy = np.sum(intersection) / np.sum(np.logical_or(mask1, mask2))
    return accuracy


def calculate_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    iou = calculate_iou(mask1, mask2)
    dice = calculate_dice(mask1, mask2)
    precision = calculate_precision(mask1, mask2)
    recall = calculate_recall(mask1, mask2)
    f1_score = calculate_f1_score(mask1, mask2)
    accuracy = calculate_accuracy(mask1, mask2)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def load_image(image_path, size, device):
    images = image_path if type(image_path) == list else [image_path]
    image_list = []
    for image in images:
        rgb_img = cv2.imread(image, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, size)
        raw_image = rgb_img.copy()
        rgb_img = np.float32(rgb_img) / 255
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4762, 0.3054, 0.2368],
                                 [0.3345, 0.2407, 0.2164])
        ])
        image_list.append((cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), transforms.ToTensor()(rgb_img).unsqueeze(0).to(device)))

    return image_list if type(image_path) == list else image_list[0]


from PIL import Image
def load_imagenet_image(image_path, device):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    raw_image = rgb_img.copy()
    rgb_img = np.float32(rgb_img) / 255
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), preprocess(input_image).unsqueeze(0).to(device)


def load_mask(mask_path, size):
    if 'kvasir-instrument' in mask_path:
        mask_path = str(Path(mask_path).with_suffix('.png'))

    rgb_mask = cv2.imread(mask_path, 1)[:, :, ::-1]
    rgb_mask = cv2.resize(rgb_mask, size)
    gray_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    gray_mask_reshaped = np.reshape(gray_mask, (1, size[0], size[1]))
    gray_mask_reshaped = np.float32(gray_mask_reshaped) / 255
    rgb_mask = np.float32(rgb_mask) / 255
    return gray_mask_reshaped, rgb_mask


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def find_bounding_box(mask, threshold):
    filtered_mask = mask > threshold
    reshaped_filtered_mask = np.expand_dims(filtered_mask, axis=0)

    # Create input_box in xyxy format from grayscale_cam
    return sv.detection.utils.mask_to_xyxy(reshaped_filtered_mask)


def distribute_points_in_bbox_grid_based(bbox, num_points):
    # Calculate the center of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Calculate the width and height of the bounding box
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Calculate the step size for x and y coordinates
    step_size_x = width / np.sqrt(num_points)
    step_size_y = height / np.sqrt(num_points)

    # Initialize the list of coordinates with the center of the bounding box
    coords = [(center_x, center_y)]

    # Generate the coordinates in a grid-like pattern
    for i in range(1, int(np.sqrt(num_points))):
        for j in range(-i, i+1):
            x = center_x + j * step_size_x
            y = center_y + i * step_size_y
            # Ensure the point is within the bounding box
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                coords.append((x, y))

    return coords


def expand_bounding_box_for_mask(input_box, binary_mask, offset=10):
    """
    Expands the bounding box if the mask is hugging or going past the barrier.

    Parameters:
    input_box (np.ndarray): The bounding box to expand.
    binary_mask (np.ndarray): The binary mask.
    offset (int): The number of pixels to expand the bounding box by.

    Returns:
    np.ndarray: The expanded bounding box.
    """
    # Get the coordinates of the non-zero values in the binary mask
    mask_coords = np.argwhere(binary_mask[0] > 0)

    # Get the min and max coordinates of the mask
    mask_min_coords = np.min(mask_coords, axis=0)
    mask_max_coords = np.max(mask_coords, axis=0)

    # Check if the mask coordinates are on the edge of the bounding box or outside it
    if mask_min_coords[0] <= input_box[1] + offset:
        # The mask is on the top edge of the bounding box or outside it
        input_box[1] = max(mask_min_coords[0] - offset, 0)
    if mask_max_coords[0] >= input_box[3] - offset:
        # The mask is on the bottom edge of the bounding box or outside it
        input_box[3] = min(mask_max_coords[0] + offset, binary_mask.shape[1] - 1)
    if mask_min_coords[1] <= input_box[0] + offset:
        # The mask is on the left edge of the bounding box or outside it
        input_box[0] = max(mask_min_coords[1] - offset, 0)
    if mask_max_coords[1] >= input_box[2] - offset:
        # The mask is on the right edge of the bounding box or outside it
        input_box[2] = min(mask_max_coords[1] + offset, binary_mask.shape[2] - 1)

    input_box[0] = max(0, input_box[0])
    input_box[1] = max(0, input_box[1])
    input_box[2] = min(binary_mask.shape[2] - 1, input_box[2])
    input_box[3] = min(binary_mask.shape[1] - 1, input_box[3])

    return input_box


def find_grad_cam_path(image_path, dataset_roots):
    original_path = Path(image_path)

    dataset_root = None
    for part in original_path.parts:
        if part in dataset_roots:
            dataset_root = original_path.parents[original_path.parts.index(part)]
            break

    if dataset_root is None:
        raise ValueError("Dataset root not found in the provided path.")

    return Config.generated_gradcam_dir / original_path.relative_to(dataset_root)


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