'''
models.py
This module provides helper functions to load and initialize models used for segmentation and classification tasks. It includes functions to load:
- A SAM mask generator for automatic mask generation.
- A mobile SAM predictor for lightweight segmentation.
- A SAM predictor for interactive segmentation.
- An image classifier based on DenseNet121 with a custom classifier layer.

Usage:
- Import the module and call the appropriate function to load a model.
- For segmentation, use:
  • load_segment_anything(model_type, checkpoint_folder, device)
  • load_mobile_segment_anything_predictor(model_type, checkpoint_folder, device)
  • load_segment_anything_predictor(model_type, checkpoint_folder, device)
- For classification, call:
  • load_image_classifier(checkpoint_path)

Steps:
1. The SAM functions load models from predefined registries using the specified checkpoint paths and move them to the provided device.
2. For the mobile SAM predictor, the corresponding registry and predictor class are used.
3. The image classifier loader creates a DenseNet121 model, replaces its classifier with a custom layer for 22 classes, loads the checkpoint weights, and sets the model to evaluation mode.

Outputs:
- Each function returns an initialized model instance ready for inference.

Dependencies:
- Standard library: pathlib
- External libraries: torch, torchvision, torch.nn
- Segmentation libraries: segment_anything, mobile_sam
'''

from pathlib import Path

import torch
import torchvision
from torch import nn

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from mobile_sam import SamPredictor as MobileSamPredictor, sam_model_registry as mobile_sam_model_registry


def load_segment_anything(model_type, checkpoint_folder, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def load_mobile_segment_anything_predictor(model_type, checkpoint_folder, device):
    sam = mobile_sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    mask_generator = MobileSamPredictor(sam)
    return mask_generator


def load_segment_anything_predictor(model_type, checkpoint_folder, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def load_image_classifier(checkpoint_path):
    # Load model
    model_path = Path(checkpoint_path)

    number_of_classes = 22

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model if already created
    model = torchvision.models.densenet121(weights=None).to(device)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, number_of_classes),
        nn.LogSoftmax(dim=1))

    checkpoint = torch.load(model_path, map_location=device)  # loading best model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

