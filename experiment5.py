'''
experiment5.py
This script performs hyperparameter tuning for generating segmentation masks using GradCAM methods. It integrates a mobile SAM predictor and an image classifier to generate class activation maps (CAMs), then optimizes parameters via scikit-optimize by evaluating the generated masks against ground truth.

Usage:
- Ensure the image and mask directories are correctly set in `IMAGES_PATH` and `MASK_PATH`.
- Verify that the necessary checkpoints for the image classifier and mobile SAM predictor are available.
- Adjust parameters such as `image_size`, `class_target`, `steps`, and `batch_size` as needed.
- Run the script to perform parameter tuning, generate CAMs, derive segmentation masks, and compute evaluation metrics.

Steps:
1. Create experiment and GradCAM output directories.
2. Load images and ground truth masks.
3. Define the search space for hyperparameters (CAM method, eigen_smooth, aug_smooth, threshold).
4. For each set of parameters, load or generate CAMs via a subprocess call to an external script.
5. Generate segmentation masks from CAMs using the mobile SAM predictor.
6. Calculate evaluation metrics (e.g., IoU) comparing generated masks with ground truth.
7. Save step-wise metrics, combined statistics, and hyperparameter settings to disk.
8. Use gp_minimize to find the best hyperparameters based on the IoU score.

Outputs:
- Generated GradCAMs stored in the designated directory.
- CSV files with per-step metrics and combined statistics.
- A JSON file summarizing hyperparameters and optimization metadata.

Dependencies:
- Standard libraries: json, os, subprocess, time, pathlib
- External libraries: pandas, torch, numpy, pytorch_grad_cam, scikit-optimize, tqdm
- Custom modules: models (load_image_classifier, load_mobile_segment_anything_predictor), utils (load_image, find_bounding_box, load_mask, store_gradcam, get_stored_gradcam, calculate_all_metrics, find_grad_cam_path, calculate_statistics, create_experiment_folder, ProgressCounterCallback)
'''

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch

import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, HiResCAM, EigenCAM, EigenGradCAM, \
    AblationCAM, ScoreCAM, XGradCAM, LayerCAM, FullGrad
from skopt import gp_minimize
from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback

from tqdm import tqdm

from models import load_image_classifier, load_mobile_segment_anything_predictor

from utils import load_image, find_bounding_box, load_mask, store_gradcam, get_stored_gradcam, \
    calculate_all_metrics, find_grad_cam_path, calculate_statistics, create_experiment_folder, ProgressCounterCallback

image_size = (224, 224)
class_target = 5

steps = 10
batch_size = 128

IMAGES_PATH = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/images')
MASK_PATH = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/masks')

experiments_dir = Path('/mnt/e/SAMexperiments/experiments')
generated_gradcam_dir = Path('/mnt/e/SAMexperiments/generated_gradcams')

experiments_dir.mkdir(exist_ok=True)
generated_gradcam_dir.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = load_mobile_segment_anything_predictor('vit_t', 'checkpoints/mobile_sam.pt', device=device)

cam_methods = {
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

image_classifier_model_path = Path('checkpoints', 'model_epochs47_batch32.pth')
image_classifier = load_image_classifier(image_classifier_model_path)

python_path = '/mnt/e/SAMexperiments/venv_wsl2/bin/python'


def run_optimizer_tuning():
    experiment_folder = create_experiment_folder(experiments_dir, 'parameter_tuning')

    step_callback = ProgressCounterCallback()
    enabled_cam_methods = cam_methods

    # Loading all the images
    print('Loading images')
    image_folder = IMAGES_PATH
    all_loaded_images = load_image([str(image_folder / image) for image in os.listdir(image_folder)], image_size, device)
    image_name_list = os.listdir(image_folder)

    # Load the ground truth masks
    print('Loading ground truth masks')
    masks_folder = MASK_PATH
    ground_truth_masks = [load_mask(str(masks_folder / Path(image_path).name), image_size)[0][0] for
                          image_path in os.listdir(image_folder)]

    for cam_method in enabled_cam_methods:
        for eigen_smooth in [True, False]:
            for aug_smooth in [True, False]:
                print(
                    f'Running optimization for {cam_method} with eigen_smooth={eigen_smooth} and aug_smooth={aug_smooth}')
                cam_descriptor = f'{cam_method}_eigen={eigen_smooth}_aug={aug_smooth}'

                cam_experiment_folder = Path(experiment_folder, cam_descriptor)
                cam_experiment_folder.mkdir(exist_ok=True)

                # Generate all cams for the dataset
                grayscale_images = np.empty((0, *image_size))

                # Load all saved gradcams
                print('Loading CAMs')
                need_cams_images = image_name_list.copy()
                first_image = image_name_list[0]
                grad_cam_base = find_grad_cam_path(str(image_folder / first_image), str(image_folder.parent.name), generated_gradcam_dir).parent / cam_descriptor
                for image_index, image in tqdm(enumerate(all_loaded_images), total=len(all_loaded_images)):
                    grayscale_cam = get_stored_gradcam(grad_cam_base, Path(image_name_list[image_index]))
                    if grayscale_cam is not None:
                        grayscale_images = np.concatenate([grayscale_images, np.expand_dims(grayscale_cam, axis=0)],
                                                          axis=0)
                        need_cams_images.remove(image_name_list[image_index])

                if len(need_cams_images) > 0:
                    for image_index in range(0, len(os.listdir(image_folder)), batch_size):
                        batch_indexes = list(
                            range(image_index, min(image_index + batch_size, len(os.listdir(image_folder)))))
                        print(f'Running batch {image_index} to {batch_indexes[-1]}')
                        batch_needs_cams = []
                        for index in batch_indexes:
                            if image_name_list[index] in need_cams_images:
                                batch_needs_cams.append(image_name_list[index])

                        if len(batch_indexes) > 0:
                            print([python_path, 'generate_gradcams.py', '--cam_method',
                                   cam_method, '--model_path', str(image_classifier_model_path),
                                   '--image_list', *[str(image_folder / image) for image in batch_needs_cams],
                                   '--parameters',
                                   json.dumps({'eigen_smooth': bool(eigen_smooth), 'aug_smooth': bool(aug_smooth),
                                               'number_of_classes': 22, 'class_target': class_target}),
                                   '--output', cam_experiment_folder])

                            result = subprocess.run(
                                [python_path, 'generate_gradcams.py', '--cam_method',
                                 cam_method, '--model_path', str(image_classifier_model_path),
                                 '--image_list', *[str(image_folder / image) for image in batch_needs_cams],
                                 '--parameters',
                                 json.dumps({'eigen_smooth': bool(eigen_smooth), 'aug_smooth': bool(aug_smooth),
                                             'number_of_classes': 22, 'class_target': class_target}),
                                 '--output', cam_experiment_folder],
                                capture_output=True,
                                text=True
                            )

                            if result.returncode != 0:
                                raise RuntimeError(f'Error running script: {result.stderr}')

                            grayscales = np.load(Path(cam_experiment_folder, 'cam.npy'))
                            grayscale_images = np.concatenate([grayscale_images, grayscales], axis=0)

                print('Finished loading CAM and images')
                # Save grayscale images to disk
                if len(need_cams_images) > 0:
                    for cam_index, grayscale_cam in enumerate(grayscale_images):
                        store_gradcam(image_folder / os.listdir(image_folder)[cam_index], cam_descriptor, grayscale_cam)

                search_space = [
                    Real(0, 1, name='threshold'),
                ]

                @use_named_args(search_space)
                def optimization_objective(threshold):
                    start_time = time.time()

                    # Generate masks for the dataset
                    print('Generating masks from CAMs')
                    masks = []
                    for cam_index, grayscale_cam in tqdm(enumerate(grayscale_images), total=len(grayscale_images)):
                        bounding_box = find_bounding_box(grayscale_cam, threshold)
                        predictor.set_image(all_loaded_images[cam_index][1])
                        mask_bounding_box, _, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=bounding_box,
                            multimask_output=False,
                        )
                        masks.append(mask_bounding_box)

                    print('Calculating metrics for CAM masks - Time elapsed: ', time.time() - start_time)
                    step_metrics = {}
                    # Calculate all metrics
                    for index, mask in tqdm(enumerate(masks), total=len(masks)):
                        gt_mask = (ground_truth_masks[index] > 0.5).ravel()
                        mask = mask[0].ravel()
                        step_metrics[os.listdir(image_folder)[index]] = calculate_all_metrics(gt_mask, mask)

                    all_statistics = calculate_statistics(step_metrics)
                    # Save metrics to disk
                    Path(cam_experiment_folder, f'step_{step_callback.current_step}').mkdir(exist_ok=True)
                    pd.DataFrame(step_metrics).to_csv(
                        Path(cam_experiment_folder, f'step_{step_callback.current_step}', f'metrics.csv'))
                    all_statistics.to_csv(
                        Path(cam_experiment_folder, f'step_{step_callback.current_step}', f'combined_metrics.csv'))

                    hyperparameters = {
                        'cam_method': cam_method,
                        'eigen_smooth': bool(eigen_smooth),
                        'aug_smooth': bool(aug_smooth),
                        'threshold': threshold,
                        'step': step_callback.current_step,
                        'batch_size': batch_size,
                    }
                    with open(Path(cam_experiment_folder, f'step_{step_callback.current_step}', 'hyperparameters.json'),
                              'w') as f:
                        json.dump(hyperparameters, f)

                    return -all_statistics['IoU / Jaccard\'s index'].iloc[0]

                res_gp = gp_minimize(optimization_objective, search_space, n_calls=steps,
                                     callback=[step_callback, VerboseCallback(steps)], verbose=True)
                sanitize_parameters = []
                for param_step in res_gp.x_iters:
                    sanitize_parameters.append(
                        [param if type(param) != np.bool_ else bool(param) for param in param_step])
                steps_metadata = {
                    'steps_ious': (-res_gp.func_vals).tolist(),
                    'steps_parameters': sanitize_parameters,
                    'steps_parameters_names': ['CAM Method', 'Eigen Smoothing', 'Augmentation Smoothing',
                                               'Bounding Box Threshold'],
                    'best_value': -res_gp.fun,
                    'best_parameters': [param if type(param) != np.bool_ else bool(param) for param in res_gp.x],
                    'best_step': int(np.argmax(-res_gp.func_vals))
                }

                with open(Path(experiment_folder, 'steps_metadata.json'), 'w') as f:
                    json.dump(steps_metadata, f)

                print('Finished optimization')


if __name__ == '__main__':
    run_optimizer_tuning()