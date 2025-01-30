import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import numpy as np
import torchvision
from pytorch_grad_cam import FullGrad, LayerCAM, XGradCAM, ScoreCAM, AblationCAM, EigenGradCAM, EigenCAM, HiResCAM, \
    GradCAMElementWise, GradCAMPlusPlus, GradCAM
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback
from torch import nn
from tqdm import tqdm

from config import Config
from models import load_mobile_segment_anything_predictor
from utils import load_image, find_bounding_box, load_mask, store_gradcam, get_stored_gradcam, calculate_all_metrics, find_grad_cam_path, calculate_statistics


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


def run_optimizer_tuning(tuning_parameters):
    experiment_folder = create_experiment_folder(Config.experiments_dir, 'parameter_tuning')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_callback = ProgressCounterCallback()

    steps = tuning_parameters['steps']
    dataset_path = Config.available_datasets[tuning_parameters['dataset']]
    class_path = Config.allowed_classes[tuning_parameters['dataset']][0]
    class_target = Config.class_targets[tuning_parameters['dataset']][0]
    mode = tuning_parameters['mode']
    batch_size = tuning_parameters['batch_size']

    include_segment_anything = tuning_parameters['include_segment_anything']

    # Loading all the images
    print('Loading images')
    image_folder = dataset_path / class_path
    all_loaded_images = load_image([str(image_folder / image) for image in os.listdir(image_folder)], Config.image_size, device)
    image_name_list = os.listdir(image_folder)

    # Load the ground truth masks
    print('Loading ground truth masks')
    masks_folder = Config.masks_paths[tuning_parameters['dataset']][class_path]
    ground_truth_masks = [load_mask(str(masks_folder / Path(image_path).name), Config.image_size)[0][0] for
                          image_path in os.listdir(image_folder)]

    cam_method_names = tuning_parameters['cam_methods']
    enabled_cam_methods = {cam_name: Config.cam_methods[cam_name] for cam_name in cam_method_names}

    for cam_method in enabled_cam_methods:
        for eigen_smooth in [True, False]:
            for aug_smooth in [True, False]:
                print(f'Running optimization for {cam_method} with eigen_smooth={eigen_smooth} and aug_smooth={aug_smooth}')
                cam_descriptor = f'{cam_method}_eigen={eigen_smooth}_aug={aug_smooth}'

                cam_experiment_folder = Path(experiment_folder, cam_descriptor)
                cam_experiment_folder.mkdir(exist_ok=True)

                # Generate all cams for the dataset
                grayscale_images = np.empty((0, *Config.image_size))

                # Load all saved gradcams
                print('Loading CAMs')
                need_cams_images = image_name_list.copy()
                first_image = image_name_list[0]
                grad_cam_base = find_grad_cam_path(str(image_folder / first_image)).parent / cam_descriptor
                for image_index, image in tqdm(enumerate(all_loaded_images), total=len(all_loaded_images)):
                    grayscale_cam = get_stored_gradcam(grad_cam_base, Path(image_name_list[image_index]))
                    if grayscale_cam is not None:
                        grayscale_images = np.concatenate([grayscale_images, np.expand_dims(grayscale_cam, axis=0)], axis=0)
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
                            python_path = Path('E:\\SAMexperiments\\venv\\Scripts\\python.exe')

                            if os.name == 'posix':
                                python_path = '/mnt/e/SAMexperiments/venv_wsl2/bin/python'

                            print([python_path, 'generate_gradcams.py', '--cam_method',
                                   cam_method, '--model_path', str(Config.model_path),
                                   '--image_list', *[str(image_folder / image) for image in batch_needs_cams],
                                   '--parameters',
                                   json.dumps({'eigen_smooth': bool(eigen_smooth), 'aug_smooth': bool(aug_smooth),
                                               'number_of_classes': 22, 'class_target': class_target}),
                                   '--output', cam_experiment_folder])

                            result = subprocess.run(
                                [python_path, 'generate_gradcams.py', '--cam_method',
                                 cam_method, '--model_path', str(Config.model_path),
                                 '--image_list', *[str(image_folder / image) for image in batch_needs_cams], '--parameters',
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
                    Categorical(list(enabled_cam_methods.keys()), name='cam_method'),
                    Categorical([True, False], name='eigen_smooth'),
                    Categorical([True, False], name='aug_smooth'),
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
                        Config.predictor.set_image(all_loaded_images[cam_index][1])
                        mask_bounding_box, _, _ = Config.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=bounding_box,
                            multimask_output=False,
                        )
                        # mask_bounding_box = update_mask_based_on_blackish_regions(all_loaded_images[cam_index][1], mask_bounding_box, threshold=30,
                        #                                                          length=5)
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
                        'dataset': tuning_parameters['dataset'],
                        'class': class_path,
                        'include_segment_anything': include_segment_anything
                    }
                    with open(Path(cam_experiment_folder, f'step_{step_callback.current_step}', 'hyperparameters.json'),
                              'w') as f:
                        json.dump(hyperparameters, f)

                    return -all_statistics['IoU / Jaccard\'s index'].iloc[0]

                res_gp = gp_minimize(optimization_objective, search_space, n_calls=steps, callback=[step_callback, VerboseCallback(steps)], verbose=True)
                sanitize_parameters = []
                for param_step in res_gp.x_iters:
                    sanitize_parameters.append([param if type(param) != np.bool_ else bool(param) for param in param_step])
                steps_metadata = {
                    'steps_ious': (-res_gp.func_vals).tolist(),
                    'steps_parameters': sanitize_parameters,
                    'steps_parameters_names': ['CAM Method', 'Eigen Smoothing', 'Augmentation Smoothing', 'Bounding Box Threshold'],
                    'best_value': -res_gp.fun,
                    'best_parameters': [param if type(param) != np.bool_ else bool(param) for param in res_gp.x],
                    'best_step': int(np.argmax(-res_gp.func_vals))
                }

                with open(Path(experiment_folder, 'steps_metadata.json'), 'w') as f:
                    json.dump(steps_metadata, f)

                print('Finished optimization')


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