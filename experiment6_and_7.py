import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import supervision as sv

from PIL import Image

import pandas as pd
import torch

from torchvision import datasets, transforms

import numpy as np
import torchvision
from pytorch_grad_cam import FullGrad, LayerCAM, XGradCAM, ScoreCAM, AblationCAM, EigenGradCAM, EigenCAM, HiResCAM, \
    GradCAMElementWise, GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.metrics.road import ROADCombined, ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback
from torch import nn
from tqdm import tqdm

from config import Config
from models import load_mobile_segment_anything_predictor, load_segment_anything
from system import prune_masks, find_overlap_with_iou, create_cam_and_mask, generate_cam
from utils import load_image, find_bounding_box, load_mask, store_gradcam, get_stored_gradcam, calculate_all_metrics, \
    find_grad_cam_path, calculate_statistics, get_stored_sam


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


# Define a custom callback to track the current step
class ProgressCounterCallback:
    def __init__(self):
        self.current_step = 0

    def __call__(self, res):
        self.current_step += 1
        #print(f"Current step: {self.current_step}")


def find_overlap(masks, predictor_mask):
    # Go through masks and pick the mask with the highest overlap with predictor_mask
    best_mask = None
    best_iou = 0

    for mask in masks:
        # Calculate the overlap between the mask and the predictor_mask
        intersection = np.sum(mask['segmentation'] & predictor_mask)
        union = np.sum(mask['segmentation'] | predictor_mask)
        iou = intersection / union

        # If the overlap is higher than the best_overlap, update the best_mask and best_overlap
        if iou > best_iou:
            best_mask = mask
            best_iou = iou

    return best_mask, best_iou



def run_optimizer_tuning(tuning_parameters):
    experiment_folder = create_experiment_folder(Config.experiments_dir, 'overlay_test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_callback = ProgressCounterCallback()

    steps = tuning_parameters['steps']
    images_path = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/images')
    gt_mask_path = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/masks')
    class_target = 5
    image_name_list = os.listdir(images_path)

    sam_masks_path = Path('/mnt/e/SAMexperiments/sam_masks/raw')

    # Loading all the images
    #print('Loading images')
    #image_folder = dataset_path / class_path
    #all_loaded_images = load_image([str(image_folder / image) for image in os.listdir(image_folder)], Config.image_size, device)


    # Load the ground truth masks
    #print('Loading ground truth masks')
    #masks_folder = Config.masks_paths[tuning_parameters['dataset']][class_path]
    #ground_truth_masks = [load_mask(str(masks_folder / Path(image_path).name), Config.image_size)[0][0] for
    #                      image_path in os.listdir(image_folder)]

    #cam_method = 'ScoreCAM'
    #eigen_smooth = False
    #aug_smooth = True
    #threshold = 0.6156759427375402

    cam_method = 'ScoreCAM'
    eigen_smooth = False
    aug_smooth = False
    threshold = 0.6469470655354411

    cam_descriptor = f'{cam_method}_eigen={eigen_smooth}_aug={aug_smooth}'
    cam_experiment_folder = Path(experiment_folder, cam_descriptor)
    cam_experiment_folder.mkdir(exist_ok=True)

    # Generate all cams for the dataset
    grayscale_images = np.empty((0, *Config.image_size))

    # Load all saved gradcams
    print('Loading CAMs')
    first_image = image_name_list[0]
    grad_cam_base = find_grad_cam_path(str(images_path / first_image)).parent / cam_descriptor

    metrics = {}
    metrics_vis = {}
    metrics_no_overlap = {}
    metrics_combined = {}
    for image_index, image_name in tqdm(enumerate(image_name_list), total=len(image_name_list)):
        start_time = time.time()
        image = Image.open(images_path / image_name)
        image = image.resize(Config.image_size)
        image_np = np.array(image)
        truth_mask = Image.open(gt_mask_path / image_name)
        truth_mask = truth_mask.resize(Config.image_size)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5
        grayscale_cam = get_stored_gradcam(grad_cam_base, Path(image_name_list[image_index]))

        print('Generating masks from CAMs')
        bounding_box = find_bounding_box(grayscale_cam, threshold)
        Config.predictor.set_image(image_np)
        mask_bounding_box, _, _ = Config.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box,
            multimask_output=False,
        )
        #mask_bounding_box = update_mask_based_on_blackish_regions(all_loaded_images[cam_index][1], mask_bounding_box, threshold=30,
        #                                                         length=5)

        #sam_mask_image = np.load((sam_masks_path / image_name).with_suffix('.npy'))
        #sam_mask_image = get_stored_sam(images_path / image_name)

        mask_generator = load_segment_anything('vit_h', 'checkpoints/sam_vit_h_4b8939.pth', device=device)
        sam_masks = mask_generator.generate(image_np)

        sam_detections = sv.Detections.from_sam(sam_masks)
        generated_sam_image = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(
            cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB), sam_detections)

        print('Finished loading CAM and images')

        image_area = image_np.shape[0] * image_np.shape[1]
        min_percentage, max_percentage = (1, 70)  # Minimum and maximum mask size (% of the image area)
        min_size = min_percentage / 100 * image_area  # Minimum acceptable mask size (% of the image area)
        max_size = max_percentage / 100 * image_area  # Maximum acceptable mask size (% of the image area)

        # Prune masks that are too big or too small
        sam_masks = [mask for mask in sam_masks if min_size < mask['area'] < max_size]

        sam_masks = prune_masks(sam_masks, image_np, threshold=30, blackish_threshold=0.5)

        sam_detections = sv.Detections.from_sam(sam_masks)
        generated_sam = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(
            cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB), sam_detections)

        #sam_masks = [mask['segmentation'] for mask in sam_masks]

        best_mask, best_iou = find_overlap(sam_masks, mask_bounding_box[0])

        if best_iou > 0.1:
            best_mask = best_mask['segmentation']
            used_overlay = 1
        else:
            best_mask = mask_bounding_box[0]
            best_iou = 0.0
            used_overlay = 0

        metrics[image_name] = calculate_all_metrics(truth_mask.ravel(), best_mask.ravel())
        metrics[image_name]['time'] = time.time() - start_time
        metrics[image_name]['best_overlap_iou'] = best_iou
        metrics[image_name]['used_overlay'] = used_overlay

        # Calculate ROAD Combined
        cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
        metric_targets = [ClassifierOutputSoftmaxTarget(5)]
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4762, 0.3054, 0.2368],
                                 [0.3345, 0.2407, 0.2164])
        ])
        input_tensor = transforms.ToTensor()(image_np).unsqueeze(0).to(device)
        scores = cam_metric(input_tensor, np.array([grayscale_cam]), metric_targets, Config.model)
        metrics[image_name]['ROAD Combined'] = scores[0]

        start_time = time.time()
        cam_metric = ROADMostRelevantFirst(percentile=80)
        scores, visualizations = cam_metric(input_tensor, np.expand_dims(grayscale_cam, axis=0), metric_targets,
                                            Config.model, return_visualization=True)
        #score = scores[0]
        visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
        visualization = np.clip(visualization, 0, 1)
        # visualization = deprocess_image(visualization)

        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4762, 0.3054, 0.2368],
                                 [0.3345, 0.2407, 0.2164])
        ])
        input_tensor = transforms.ToTensor()(visualization).unsqueeze(0).to(device)

        #cam_and_mask_return = create_cam_and_mask(image_np, cv2.cvtColor(visualization.copy(), cv2.COLOR_BGR2RGB),
        #                                          input_tensor, Config.cam_methods[cam_method], class_target, eigen_smooth,
        #                                          aug_smooth, 30, True, True,
        #                                          0, None,
        #                                          0, False)

        #vis_grayscale_cam, vis_heatmap, vis_bounding_box_drawn_on_image, vis_mask_image, vis_mask_bounding_box, vis_bounding_box = cam_and_mask_return

        # Generate class activation map
        with Config.cam_methods[cam_method](model=Config.model, target_layers=Config.target_layers) as cam:
            grayscale_cam, heatmap = generate_cam(cam, visualization, input_tensor,
                                                  [ClassifierOutputTarget(class_target)], eigen_smooth, aug_smooth)

        print('Generating masks from visualization CAMs')
        bounding_box = find_bounding_box(grayscale_cam, threshold)
        Config.predictor.set_image(visualization)
        mask_bounding_box, _, _ = Config.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box,
            multimask_output=False,
        )

        metrics_no_overlap[image_name] = calculate_all_metrics(truth_mask.ravel(), mask_bounding_box[0].ravel())
        metrics_no_overlap[image_name]['time'] = time.time() - start_time

        best_vis_mask, best_iou = find_overlap(sam_masks, mask_bounding_box[0])

        if best_iou > 0.1:
            best_vis_mask = best_vis_mask['segmentation']
            used_overlay = 1
        else:
            best_vis_mask = mask_bounding_box[0]
            best_iou = 0.0
            used_overlay = 0

        metrics_vis[image_name] = calculate_all_metrics(truth_mask.ravel(), best_vis_mask.ravel())
        metrics_vis[image_name]['time'] = time.time() - start_time
        metrics_vis[image_name]['best_overlap_iou'] = best_iou
        metrics_vis[image_name]['used_overlay'] = used_overlay

        # Calculate ROAD Combined
        cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
        metric_targets = [ClassifierOutputSoftmaxTarget(5)]
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4762, 0.3054, 0.2368],
                                 [0.3345, 0.2407, 0.2164])
        ])
        input_tensor = transforms.ToTensor()(image_np).unsqueeze(0).to(device)
        scores = cam_metric(input_tensor, np.array([grayscale_cam]), metric_targets, Config.model)
        metrics_vis[image_name]['ROAD Combined'] = scores[0]

        # Combine the masks
        best_mask = best_mask | best_vis_mask
        metrics_combined[image_name] = calculate_all_metrics(truth_mask.ravel(), best_mask.ravel())

    statistics = calculate_statistics(metrics)

    pd.DataFrame(metrics).to_csv(cam_experiment_folder / 'overlay_metrics.csv')
    statistics.to_csv(cam_experiment_folder / 'overlay_statistics.csv')

    statistics = calculate_statistics(metrics_vis)

    pd.DataFrame(metrics_vis).to_csv(cam_experiment_folder / 'vis_overlay_metrics.csv')
    statistics.to_csv(cam_experiment_folder / 'vis_overlay_statistics.csv')

    statistics = calculate_statistics(metrics_no_overlap)

    pd.DataFrame(metrics_no_overlap).to_csv(cam_experiment_folder / 'vis_metrics.csv')
    statistics.to_csv(cam_experiment_folder / 'vis_statistics.csv')

    statistics = calculate_statistics(metrics_combined)

    pd.DataFrame(metrics_combined).to_csv(cam_experiment_folder / 'combined_metrics.csv')
    statistics.to_csv(cam_experiment_folder / 'combined_statistics.csv')


if __name__ == '__main__':
    current_tuning_parameters = {'dataset': 'Kvasir-SEG',
                                 'cam_methods': ['GradCAM++', 'GradCAM', 'GradCAMElementWise', 'HiResCAM', 'EigenCAM', 'EigenGradCAM', 'AblationCAM', 'ScoreCAM', 'XGradCAM', 'LayerCAM', 'FullGrad'],
                                 'eigen_smooth': True,
                                 'aug_smooth': True,
                                 'expand_bounding_box': False,
                                 'point_strategy': 'Max Value',
                                 'threshold': 0.3,
                                 'enabled_cam_methods': [],
                                 'mode': 'Bayesian Optimization',
                                 'steps': 15,
                                 'batch_size': 8,
                                 'include_segment_anything': False,
                                 }

    # Load model
    model_path = Path('checkpoints', 'model_epochs47_batch32.pth')
    Config.add(model_path)
    number_of_classes = 22

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.add(device)

    # Load model if already created
    model = torchvision.models.densenet121(weights=None).to(device)
    Config.add(model)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, number_of_classes),
        nn.LogSoftmax(dim=1))

    checkpoint = torch.load(model_path, map_location=device)  # loading best model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load CAM methods
    target_layers = [model.features[-1]]
    Config.add(target_layers)
    '''
    Config.cam_methods = {
        'GradCAM': GradCAM(model=model, target_layers=target_layers),
        'GradCAM++': GradCAMPlusPlus(model=model, target_layers=target_layers),
        'GradCAMElementWise': GradCAMElementWise(model=model, target_layers=target_layers),
        'HiResCAM': HiResCAM(model=model, target_layers=target_layers),
        'EigenCAM': EigenCAM(model=model, target_layers=target_layers),
        'EigenGradCAM': EigenGradCAM(model=model, target_layers=target_layers),
        'AblationCAM': AblationCAM(model=model, target_layers=target_layers),
        'ScoreCAM': ScoreCAM(model=model, target_layers=target_layers),
        'XGradCAM': XGradCAM(model=model, target_layers=target_layers),
        'LayerCAM': LayerCAM(model=model, target_layers=target_layers),
        'FullGrad': FullGrad(model=model, target_layers=target_layers)
    }
    '''

    Config.cam_methods = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'GradCAMElementWise': GradCAMElementWise,
        'HiResCAM': HiResCAM,
        'EigenCAM': EigenCAM,
        'EigenGradCAM': EigenGradCAM,
        'AblationCAM': AblationCAM,
        'ScoreCAM': ScoreCAM,
        'XGradCAM': XGradCAM,
        'LayerCAM': LayerCAM,
        'FullGrad': FullGrad
    }

    Config.predictor = load_mobile_segment_anything_predictor('vit_t', 'checkpoints/mobile_sam.pt', device=device)

    Config.dataset_roots = [
        'Kvasir-SEG',
        'kvasir-instrument',
        'kvasir-sessile',
        'gastrovision_split'
    ]

    if os.name == 'posix':
        available_datasets = {
            'Kvasir-SEG': Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG'),
            'Kvasir-Instrument': Path('/mnt/e/Datasets/kvasir-instrument'),
            'Kvasir-SEG-Sessile': Path('/mnt/e/Datasets/kvasir-sessile/sessile-main-Kvasir-SEG'),
            'Gastrovision-Split-Test': Path('/mnt/e/SAMexperiments/datasets/gastrovision_split/test'),
            'Gastrovision-Split-Train': Path('/mnt/e/SAMexperiments/datasets/gastrovision_split/train')
        }
    else:
        available_datasets = {
            'Kvasir-SEG': Path('E:\\Datasets\\kvasir-seg\\Kvasir-SEG'),
            'Kvasir-Instrument': Path('E:\\Datasets\\kvasir-instrument'),
            'Kvasir-SEG-Sessile': Path('E:\\Datasets\\kvasir-sessile\\sessile-main-Kvasir-SEG'),
            'Gastrovision-Split-Test': Path('E:\\SAMexperiments\\datasets\\gastrovision_split\\test'),
            'Gastrovision-Split-Train': Path('E:\\SAMexperiments\\datasets\\gastrovision_split\\train')
        }
    Config.add(available_datasets)

    Config.allowed_classes = {
        'Kvasir-SEG': ['images'],
        'Kvasir-Instrument': ['images'],
        'Kvasir-SEG-Sessile': ['images'],
        'Gastrovision-Split-Test': os.listdir(available_datasets['Gastrovision-Split-Test']),
        'Gastrovision-Split-Train': os.listdir(available_datasets['Gastrovision-Split-Train']),
    }

    if os.name == 'posix':
        Config.masks_paths = {
            'Kvasir-SEG': {
                'images': Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG') / 'masks'
            },
            'Kvasir-Instrument': {
                'images': Path('/mnt/e/Datasets/kvasir-instrument') / 'masks'
            },
            'Kvasir-SEG-Sessile': {
                'images': Path('/mnt/e/Datasets/kvasir-sessile/sessile-main-Kvasir-SEG') / 'masks'
            },
            'Gastrovision-Split-Test': {class_name: None for class_name in
                                        os.listdir(available_datasets['Gastrovision-Split-Test'])},
            'Gastrovision-Split-Train': {class_name: None for class_name in
                                         os.listdir(available_datasets['Gastrovision-Split-Train'])}
        }
    else:
        Config.masks_paths = {
            'Kvasir-SEG': {
                'images': Path('E:\\Datasets\\kvasir-seg\\Kvasir-SEG') / 'masks'
            },
            'Kvasir-Instrument': {
                'images': Path('E:\\Datasets\\kvasir-instrument') / 'masks'
            },
            'Kvasir-SEG-Sessile': {
                'images': Path('E:\\Datasets\\kvasir-sessile\\sessile-main-Kvasir-SEG') / 'masks'
            },
            'Gastrovision-Split-Test': {class_name: None for class_name in
                                        os.listdir(available_datasets['Gastrovision-Split-Test'])},
            'Gastrovision-Split-Train': {class_name: None for class_name in
                                         os.listdir(available_datasets['Gastrovision-Split-Train'])}
        }

    Config.class_targets = {
        'Kvasir-SEG': [5],
        'Kvasir-Instrument': [0],
        'Kvasir-SEG-Sessile': [5],
        'Gastrovision-Split-Test': list(range(len(os.listdir(available_datasets['Gastrovision-Split-Test'])))),
        'Gastrovision-Split-Train': list(range(len(os.listdir(available_datasets['Gastrovision-Split-Train']))))
    }

    Config.image_size = (224, 224)
    Config.number_of_images = 6

    # Human-readable descriptions of the parameters
    Config.parameter_descriptions = {
        'dataset': 'Dataset. Choose the dataset to optimize.',
        'cam_methods': 'CAM Methods. List of CAM methods available to optimize.',
        'eigen_smooth': 'Eigen Smooth. This has the effect of removing a lot of noise.',
        'aug_smooth': 'Augmentation smoothing. This has the effect of better centering the CAM around the objects.',
        'expand_bounding_box': 'Expand Bounding Box. If the mask is crossing or reaching the bounding box, expand it.',
        'point_strategy': 'Point Strategy. Choose between Grid Based and Max Value Based.',
        'threshold': 'Bounding Box Threshold. Decides from which heatmap value to consider as a bounding box. '
                     'Between 0 and 1.',
        'enabled_cam_methods': 'Enabled CAM Methods. Selected CAM methods for optimization.',
        'mode': 'Parameter optimization mode. Choose between Bayesian Optimization and Random Search',
        'steps': 'Steps. Number of steps to run the optimization.',
        'batch_size': 'Batch Size. Number of images to process in a batch for CAM generation.',
    }

    if os.name == 'posix':
        Config.experiments_dir = Path('/mnt/e/SAMexperiments/experiments')
        Config.generated_gradcam_dir = Path('/mnt/e/SAMexperiments/generated_gradcams')
    else:
        Config.experiments_dir = Path('E:\\SAMexperiments\\experiments')
        Config.generated_gradcam_dir = Path('E:\\SAMexperiments\\generated_gradcams')

    Config.experiments_dir.mkdir(exist_ok=True)
    Config.generated_gradcam_dir.mkdir(exist_ok=True)

    run_optimizer_tuning(current_tuning_parameters)