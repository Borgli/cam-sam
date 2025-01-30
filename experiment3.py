import csv
import json
import os
import urllib
import time
from functools import lru_cache
from pathlib import Path
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import gradio as gr
import numpy as np
import PIL
import pandas as pd

import supervision as sv
from PIL import Image
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from sklearn.metrics import jaccard_score

from system import prune_masks

CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 1024
TOP_K_OBJ = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


@lru_cache
def load_mask_generator_predict() -> SamAutomaticMaskGenerator:
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamPredictor(sam)
    return mask_generator


@lru_cache
def load_clip(
        name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y: y + h, x: x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = PIL.Image.fromarray(crop)
    return crop


def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]


def filter_masks(
        image: np.ndarray,
        masks: List[Dict[str, Any]],
        #predicted_iou_threshold: float,
        #stability_score_threshold: float,
        query: str,
        clip_threshold: float,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
                #mask["predicted_iou"] < predicted_iou_threshold
                #or mask["stability_score"] < stability_score_threshold
                image.shape[:2] != mask["segmentation"].shape[:2]
                or query
                and get_score(crop_image(image, mask), get_texts(query)) < clip_threshold
        ):
            continue

        filtered_masks.append(mask)

    return filtered_masks


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


def segment(
        #predicted_iou_threshold: float,
        #stability_score_threshold: float,
        clip_threshold: float,
        image_path: str,
        query: str,
) -> PIL.ImageFile.ImageFile:
    mask_generator = load_mask_generator()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reduce the size to save gpu memory
    #image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    masks = filter_masks(
        image,
        masks,
        #predicted_iou_threshold,
        #stability_score_threshold,
        query,
        clip_threshold,
    )
    return masks


if __name__ == '__main__':

    Path('sam_clip').mkdir(exist_ok=True)
    Path('sam_clip', 'overlay').mkdir(exist_ok=True)
    Path('sam_clip', 'raw').mkdir(exist_ok=True)
    Path('sam_clip', 'raw_npy').mkdir(exist_ok=True)

    images_path = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/images')
    mask_path = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/masks')

    hyperparameters = {
        "predicted_iou_threshold": 0.9,
        "stability_score_threshold": 0.8,
        "clip_threshold": 0.9,
        "query": "polyp"
    }

    iou_scores_micro = []
    for image_name in os.listdir(images_path):
        image = Image.open(images_path / image_name)
        image_np = np.array(image)
        truth_mask = Image.open(mask_path / image_name)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5

        masks = segment(
            #hyperparameters["predicted_iou_threshold"],
            #hyperparameters["stability_score_threshold"],
            hyperparameters["clip_threshold"],
            str(images_path / image_name),
            hyperparameters["query"],
        )

        seg_masks = np.stack([mask['segmentation'] for mask in masks], axis=0)
        np.save(Path('sam_clip', 'raw_npy', Path(image_name).with_suffix('.npy')), seg_masks)

        image_area = image_np.shape[0] * image_np.shape[1]
        min_percentage, max_percentage = (1, 70)  # Minimum and maximum mask size (% of the image area)
        min_size = min_percentage / 100 * image_area  # Minimum acceptable mask size (% of the image area)
        max_size = max_percentage / 100 * image_area  # Maximum acceptable mask size (% of the image area)

        # Prune masks that are too big or too small
        masks = [mask for mask in masks if min_size < mask['area'] < max_size]

        masks = prune_masks(masks, image_np, threshold=30, blackish_threshold=0.5)

        sam_detections = sv.Detections.from_sam(masks)
        generated_sam = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(image_np, sam_detections)

        Image.fromarray(generated_sam).save(Path('sam_clip', 'overlay', image_name))


        masks = [mask['segmentation'] for mask in masks]
        if masks:
            seg_masks = np.stack(masks, axis=0)
            sam_mask = np.sum(seg_masks, axis=0, dtype=bool)
            Image.fromarray(sam_mask).save(Path('sam_clip', 'raw', image_name))
            iou_micro = jaccard_score(truth_mask, sam_mask, average='micro')
        else:
            Image.fromarray(np.zeros(image_np.shape)).save(Path('sam_clip', 'raw', image_name))
            iou_micro = 0.0
        iou_scores_micro.append(iou_micro)
        print(f'IoU Micro: {iou_micro}')

    df_iou_scores_micro = pd.DataFrame(iou_scores_micro)
    df_iou_scores_micro.index = os.listdir(images_path)
    df_iou_scores_micro.to_csv(Path('sam_clip', 'iou_scores_micro.csv'), index=False)

    print(f'Average IoU Micro: {np.mean(iou_scores_micro)}')