import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
import time
import numpy as np
import gradio as gr
import supervision as sv
import torch
from PIL import ImageDraw, Image, ImageFont
from numpy.ma.core import array
from torchvision import transforms
import torch.nn.functional as F

from numpy import unravel_index
from pytorch_grad_cam.metrics.road import ROADCombined, ROADMostRelevantFirst
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from transformers import AutoModelForCausalLM, AutoProcessor

from models import load_segment_anything, load_segment_anything_2, load_segment_anything_2_predictor
from utils import (
    load_image,
    find_bounding_box,
    distribute_points_in_bbox_grid_based,
    load_mask,
    calculate_metrics,
    store_gradcam,
    get_stored_gradcam,
    calculate_iou,
    load_imagenet_image
)

# If you need a dictionary of your CAM methods, define it below.
# Example:
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
# cam_methods = {
#   'GradCAM': GradCAM,
#   'GradCAM++': GradCAMPlusPlus,
#   'XGradCAM': XGradCAM,
#   'EigenCAM': EigenCAM
#   # etc.
# }


########################################
# Helper function to expand bounding box
########################################
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
    mask_coords = np.argwhere(binary_mask[0] > 0)
    mask_min_coords = np.min(mask_coords, axis=0)
    mask_max_coords = np.max(mask_coords, axis=0)

    if mask_min_coords[0] <= input_box[1] + offset:
        input_box[1] = max(mask_min_coords[0] - offset, 0)
    if mask_max_coords[0] >= input_box[3] - offset:
        input_box[3] = min(mask_max_coords[0] + offset, binary_mask.shape[1] - 1)
    if mask_min_coords[1] <= input_box[0] + offset:
        input_box[0] = max(mask_min_coords[1] - offset, 0)
    if mask_max_coords[1] >= input_box[2] - offset:
        input_box[2] = min(mask_max_coords[1] + offset, binary_mask.shape[2] - 1)

    input_box[0] = max(0, input_box[0])
    input_box[1] = max(0, input_box[1])
    input_box[2] = min(binary_mask.shape[2] - 1, input_box[2])
    input_box[3] = min(binary_mask.shape[1] - 1, input_box[3])

    return input_box

########################################
# CAM Generation
########################################
def generate_cam(cam, original_image, image_tensor, class_targets, eigen=False, aug=False):
    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=class_targets,
        eigen_smooth=eigen,
        aug_smooth=aug
    )
    grayscale_cam = grayscale_cam[0, :]
    output_on_image = show_cam_on_image(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
        grayscale_cam,
        use_rgb=True
    )
    return grayscale_cam, output_on_image

########################################
# Overlap detection
########################################
def find_overlap(masks, predictor_mask):
    best_mask = None
    best_overlap = 0
    for mask in masks:
        overlap = np.sum(mask['segmentation'] & predictor_mask)
        if overlap > best_overlap:
            best_mask = mask
            best_overlap = overlap
    return best_mask

def find_overlap_with_iou(masks, predictor_mask):
    best_mask = None
    best_iou = 0
    for mask in masks:
        intersection = np.sum(mask['segmentation'] & predictor_mask)
        union = np.sum(mask['segmentation'] | predictor_mask)
        iou = intersection / union if union > 0 else 0
        if iou > best_iou:
            best_mask = mask
            best_iou = iou
    return best_mask

########################################
# Mask corner updates
########################################
def update_mask_based_on_blackish_regions(image, mask, threshold=30, length=5):
    height, width, _ = image.shape
    updated_mask = mask.copy()

    def is_blackish(pixel):
        return np.all(pixel < threshold)

    def check_blackish_segment(segment):
        return all(is_blackish(pixel) for pixel in segment)

    for direction in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
        if direction == 'top-left':
            for i in range(height):
                for j in range(width):
                    if i + length < height and j + length < width:
                        segment = [image[i + k, j + k] for k in range(length)]
                        if check_blackish_segment(segment):
                            for k in range(length):
                                updated_mask[0, i + k, j + k] = False
                        else:
                            break
        elif direction == 'top-right':
            for i in range(height):
                for j in range(width - 1, -1, -1):
                    if i + length < height and j - length >= 0:
                        segment = [image[i + k, j - k] for k in range(length)]
                        if check_blackish_segment(segment):
                            for k in range(length):
                                updated_mask[0, i + k, j - k] = False
                        else:
                            break
        elif direction == 'bottom-left':
            for i in range(height - 1, -1, -1):
                for j in range(width):
                    if i - length >= 0 and j + length < width:
                        segment = [image[i - k, j + k] for k in range(length)]
                        if check_blackish_segment(segment):
                            for k in range(length):
                                updated_mask[0, i - k, j + k] = False
                        else:
                            break
        elif direction == 'bottom-right':
            for i in range(height - 1, -1, -1):
                for j in range(width - 1, -1, -1):
                    if i - length >= 0 and j - length >= 0:
                        segment = [image[i - k, j - k] for k in range(length)]
                        if check_blackish_segment(segment):
                            for k in range(length):
                                updated_mask[0, i - k, j - k] = False
                        else:
                            break
    return updated_mask

########################################
# Blackish pixel checks
########################################
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

def prune_masks(masks, image, threshold=30, blackish_threshold=0.5):
    pruned_masks = []
    for mask in masks:
        blackish_proportion = calculate_blackish_proportion(mask['segmentation'], image, threshold)
        if blackish_proportion < blackish_threshold:
            pruned_masks.append(mask)
    return pruned_masks

########################################
# Visualization
########################################
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return visualization

########################################
# create_cam_and_mask without config
########################################
def create_cam_and_mask(
    raw_image,
    rgb_image,
    image_tensor,
    cam_method_name,
    class_target,
    eigen,
    aug,
    threshold_value,
    remove_black_corners_mask,
    remove_black_corners_cam,
    number_of_points,
    points_distribution_method,
    points_distribution_threshold,
    expand_box,
    zero_shot_model,
    cam_methods,
    predictor,
    model,
    target_layers,
    image_size
):
    # Generate class activation map
    with cam_methods[cam_method_name](model=model, target_layers=target_layers) as cam:
        grayscale_cam, heatmap = generate_cam(
            cam,
            rgb_image,
            image_tensor,
            [ClassifierOutputTarget(class_target)],
            eigen,
            aug
        )

    # Remove black corners from the heatmap
    if remove_black_corners_cam:
        truth_mask = update_mask_based_on_blackish_regions(
            raw_image,
            np.expand_dims(grayscale_cam.copy(), axis=0).astype(bool),
            threshold=30,
            length=5
        )
        grayscale_cam[~truth_mask[0]] = 0
        heatmap = show_cam_on_image(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), grayscale_cam, use_rgb=True)

    bounding_box = find_bounding_box(grayscale_cam, threshold_value)

    # Set input points
    input_points = None
    input_labels = None
    if number_of_points:
        input_point = np.array([list(unravel_index(grayscale_cam.argmax(), grayscale_cam.shape))])
        input_point = np.flip(input_point, axis=1)
        input_labels = np.array([1])

        if points_distribution_method == 'Grid Based':
            input_points = np.array(
                distribute_points_in_bbox_grid_based(bounding_box[0], number_of_points),
                dtype=np.uint8
            )
            input_labels = np.ones(number_of_points)
        elif points_distribution_method == 'Max Value Based':
            positive_input_point = np.argwhere(grayscale_cam > points_distribution_threshold)
            positive_input_point = np.array(
                sorted(
                    list(positive_input_point),
                    key=lambda x: grayscale_cam[positive_input_point[x[0], x[0]]]
                )
            )
            if len(positive_input_point) > 0:
                if len(positive_input_point) < number_of_points:
                    number_of_points = len(positive_input_point)
                indices = np.linspace(0, len(positive_input_point) - 1, number_of_points, dtype=int)
                input_points = np.flip(positive_input_point[indices], axis=1)
                input_labels = np.ones(number_of_points)
            else:
                input_points = None
                input_labels = None

    if zero_shot_model == 'SAM':
        predictor.set_image(raw_image)
        mask_bounding_box, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=bounding_box,
            multimask_output=False,
        )
    else:
        # For example, using load_segment_anything_2_predictor
        local_predictor = load_segment_anything_2_predictor()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            local_predictor.set_image(raw_image)
            mask_bounding_box, _, _ = local_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=bounding_box,
                multimask_output=False,
            )
            mask_bounding_box = mask_bounding_box.astype(bool)

    if expand_box:
        current_input_box = bounding_box
        expanded_box = np.array([
            expand_bounding_box_for_mask(bounding_box[0], mask_bounding_box.astype(int), offset=10)
        ])
        while not (expanded_box == current_input_box).all():
            current_input_box = expanded_box
            expanded_mask_bounding_box, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=current_input_box,
                multimask_output=False,
            )
            expanded_box = np.array([
                expand_bounding_box_for_mask(current_input_box[0], expanded_mask_bounding_box, offset=10)
            ])
            bounding_box = expanded_box
            mask_bounding_box = expanded_mask_bounding_box

    if remove_black_corners_mask:
        mask_bounding_box = update_mask_based_on_blackish_regions(
            raw_image,
            mask_bounding_box,
            threshold=30,
            length=5
        )

    detections = sv.Detections(bounding_box, mask_bounding_box, class_id=np.array([1]))
    bounding_box_drawn_on_image = sv.BoundingBoxAnnotator().annotate(
        cv2.cvtColor(raw_image.copy(), cv2.COLOR_BGR2RGB),
        detections
    )

    if input_points is not None:
        for point in input_points:
            bounding_box_drawn_on_image = cv2.circle(bounding_box_drawn_on_image, point, 3, (3, 111, 252), -1)

    mask_image = sv.MaskAnnotator().annotate(bounding_box_drawn_on_image.copy(), detections)

    return grayscale_cam, heatmap, bounding_box_drawn_on_image, mask_image, mask_bounding_box, bounding_box

colormap = [
    'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
    'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue'
]

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)

    scale = 1

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()

            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def plot_bbox(image, data):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array

def run_mask_generation(
    event: gr.SelectData,
    output,
    eigen,
    aug,
    threshold_value,
    class_target,
    mask_path,
    cam_method_name,
    number_of_points,
    points_distribution_method,
    points_distribution_threshold,
    mask_filtering,
    remove_black_corners_overlay,
    remove_black_corners_mask,
    remove_black_corners_cam,
    remove_and_debias_threshold,
    expand_box,
    zero_shot_model,
    dataset_name,
    # Provide these needed parameters as well
    model,
    target_layers,
    cam_methods,
    predictor,
    image_size,
    available_class_targets
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()

    output_text_stream = '> Starting generating masks...\n'

    def output_update():
        output.update(output=output_text_stream.strip())
        return tuple(output.values())

    yield output_update()

    rgb_image, raw_image, image_tensor = load_image(event.value['image']['path'], image_size, device)

    if dataset_name == 'ImageNet':
        rgb_image, raw_image, image_tensor = load_imagenet_image(event.value['image']['path'], device)

    if mask_path:
        original_image = cv2.imread(event.value['image']['path'], cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        if mask_path.parent.name == 'kvasir-instrument':
            gray_mask = cv2.imread(str(mask_path / Path(event.value['image']['path']).with_suffix('.png').name), cv2.IMREAD_GRAYSCALE)
        else:
            gray_mask = cv2.imread(str(mask_path / Path(event.value['image']['path']).name), cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = gray_mask.copy()
        ground_truth_mask = np.float32(ground_truth_mask) / 255
        detections = sv.Detections(
            np.array([[0, 0, 0, 0]]),
            np.array([ground_truth_mask > 0.5]),
            class_id=np.array([1])
        )
        ground_truth_real_size = sv.MaskAnnotator(color=sv.Color.GREEN).annotate(original_image, detections)
        output['ground_truth_image'] = gr.Image(visible=True, value=ground_truth_real_size)
        ground_truth_mask = gray_mask.copy()
        ground_truth_mask = np.where(ground_truth_mask < 100, 0, 255)
        ground_truth_mask = ground_truth_mask.astype(np.uint8)
        ground_truth_mask = cv2.resize(ground_truth_mask, image_size)
        ground_truth_mask = np.float32(ground_truth_mask) / 255
        yield tuple(output.values())
    else:
        output['ground_truth_image'] = gr.Image(visible=False)

    if dataset_name == 'ImageNet':
        output_text_stream += '> Detected ImageNet and will run default model and inference to find class_target\n'
        yield output_update()

        model_imagenet = torchvision.models.densenet121(weights='IMAGENET1K_V1').to(device)
        model_imagenet.eval()
        result = model_imagenet(image_tensor)
        probabilities = F.softmax(result, dim=1)
        probabilities = probabilities.cpu().detach().numpy()[0]
        highest_confidence = np.argmax(probabilities)
        class_target = highest_confidence
        class_name = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1.meta['categories'][highest_confidence]
        output_text_stream += f'> Highest confident score is for class "{class_name}" with score {probabilities[highest_confidence]}\n'
    else:
        result = model(image_tensor)
        probabilities = F.softmax(result, dim=1)
        probabilities = probabilities.cpu().detach().numpy()[0]
        highest_confidence = np.argmax(probabilities)
        output_text_stream += f'> Classifier confidence for class "{available_class_targets[class_target]}" is {probabilities[class_target]}\n'
        if highest_confidence != class_target:
            output_text_stream += (f'> Note: Higher confidence was found using class ' +
                                   f'"{available_class_targets[highest_confidence]}" ' +
                                   f'with {probabilities[highest_confidence]}\n')

    output_text_stream += '> Generating CAM and mask... '
    yield output_update()

    if dataset_name == 'ImageNet':
        cam_and_mask_return = create_cam_and_mask(
            raw_image,
            rgb_image,
            image_tensor,
            cam_method_name,
            class_target,
            eigen,
            aug,
            threshold_value,
            remove_black_corners_mask,
            remove_black_corners_cam,
            number_of_points,
            points_distribution_method,
            points_distribution_threshold,
            expand_box,
            zero_shot_model,
            cam_methods,
            predictor,
            model_imagenet,
            [model_imagenet.features[-1]],
            image_size
        )
    else:
        cam_and_mask_return = create_cam_and_mask(
            raw_image,
            rgb_image,
            image_tensor,
            cam_method_name,
            class_target,
            eigen,
            aug,
            threshold_value,
            remove_black_corners_mask,
            remove_black_corners_cam,
            number_of_points,
            points_distribution_method,
            points_distribution_threshold,
            expand_box,
            zero_shot_model,
            cam_methods,
            predictor,
            model,
            target_layers,
            image_size
        )

    grayscale_cam, heatmap, bounding_box_drawn_on_image, mask_image, mask_bounding_box, bounding_box = cam_and_mask_return

    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    scores = cam_metric(image_tensor, np.expand_dims(grayscale_cam, axis=0), [ClassifierOutputSoftmaxTarget(class_target)], (model_imagenet if dataset_name=="ImageNet" else model))
    score = scores[0]

    output_text_stream += f'Done\n> ROAD Score: {score}\n> Generating masks from auto-segmentation model... '
    output.update(
        cam_image=cv2.resize(heatmap, (1024, 1024)),
        bounding_box_image=cv2.resize(bounding_box_drawn_on_image, (1024, 1024)),
        bounding_box_mask_image=cv2.resize(mask_image, (1024, 1024)),
        output=output_text_stream.strip()
    )
    yield tuple(output.values())

    if zero_shot_model == 'SAM':
        mask_generator = load_segment_anything('vit_h', 'checkpoints/sam_vit_h_4b8939.pth', device=device)
        masks = mask_generator.generate(raw_image)
    elif zero_shot_model == 'SAM-2':
        mask_generator = load_segment_anything_2()
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks = mask_generator.generate(raw_image)

    image_area = raw_image.shape[0] * raw_image.shape[1]
    min_percentage, max_percentage = mask_filtering
    min_size = min_percentage / 100 * image_area
    max_size = max_percentage / 100 * image_area

    if remove_black_corners_overlay:
        masks = prune_masks(masks, raw_image, threshold=30, blackish_threshold=0.5)

    masks = [mask for mask in masks if min_size < mask['area'] < max_size]

    sam_detections = sv.Detections.from_sam(masks)
    generated_sam = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(
        cv2.cvtColor(raw_image.copy(), cv2.COLOR_BGR2RGB), sam_detections)

    best_mask_annotated = None
    best_mask = find_overlap_with_iou(masks, mask_bounding_box[0])

    if best_mask:
        detections = sv.Detections.from_sam([best_mask])
        best_mask_annotated = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(
            cv2.cvtColor(raw_image.copy(), cv2.COLOR_BGR2RGB),
            detections
        )
        detections_2 = sv.Detections(bounding_box, mask_bounding_box)
        combined_detections = sv.Detections.merge([detections, detections_2])
        show_overlap = sv.MaskAnnotator(color_lookup=sv.annotators.utils.ColorLookup.INDEX).annotate(
            cv2.cvtColor(raw_image.copy(), cv2.COLOR_BGR2RGB),
            combined_detections
        )
    else:
        no_overlap = np.zeros((400, 600, 3), dtype=np.uint8)
        pil_image = Image.fromarray(no_overlap)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("SourceSans3-Regular.ttf", 40)
        text = "No overlapping image found"
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((600 - (bbox[2] - bbox[0])) // 2, (400 - (bbox[3] - bbox[1])) // 2), text, font=font, fill=(255, 255, 255))
        no_overlap = np.array(pil_image)
        best_mask_annotated = no_overlap
        show_overlap = no_overlap

    output_text_stream += 'Done\n'

    def find_overlap(masks, predictor_mask):
        best_mask = None
        best_iou = 0
        for mask in masks:
            intersection = np.sum(mask['segmentation'] & predictor_mask)
            union = np.sum(mask['segmentation'] | predictor_mask)
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_mask = mask
                best_iou = iou
        return best_mask, best_iou

    best_mask, best_iou = find_overlap(masks, mask_bounding_box[0])
    if best_iou > 0.1 or np.sum(mask_bounding_box[0]) < 100:
        final_output = best_mask_annotated
    else:
        detections = sv.Detections(bounding_box, mask_bounding_box, class_id=np.array([1]))
        final_output = sv.MaskAnnotator().annotate(cv2.cvtColor(raw_image.copy(), cv2.COLOR_BGR2RGB), detections)

    if mask_path:
        if best_iou > 0.1:
            ground_truth_iou = round(calculate_iou(ground_truth_mask > 0.5, best_mask['segmentation']), 2)
        else:
            ground_truth_iou = round(calculate_iou(ground_truth_mask > 0.5, mask_bounding_box[0] > 0.5), 2)
    else:
        ground_truth_iou = "No ground truth mask available"

    if dataset_name == 'ImageNet':
        parameters = {
            'cam_method_name': cam_method_name,
            'eigen_smoothing': eigen,
            'aug_smoothing': aug,
            'bounding_box_threshold': threshold_value,
            'zero_shot_model': zero_shot_model,
            'class_target': "ImageNet_class"  # or class_name if you prefer
        }
        summary = (
            f'## Result Summary\n'
            f'#### Using result: {"CAM Generated Mask" if not best_iou > 0.1 else "Best Overlapping Mask"}\n'
            f'#### IoU with ground truth mask: {ground_truth_iou}\n'
            f'#### Using Parameters: {parameters}\n'
            f'#### Time taken: {time.time() - start_time:.2f} seconds\n'
        )
    else:
        parameters = {
            'cam_method_name': cam_method_name,
            'eigen_smoothing': eigen,
            'aug_smoothing': aug,
            'bounding_box_threshold': threshold_value,
            'zero_shot_model': zero_shot_model,
            'class_target': available_class_targets[class_target]
        }
        summary = (
            f'## Result Summary\n'
            f'#### Using result: {"CAM Generated Mask" if not best_iou > 0.1 else "Best Overlapping Mask"}\n'
            f'#### IoU with ground truth mask: {ground_truth_iou}\n'
            f'#### Using Parameters: {parameters}\n'
            f'#### Time taken: {time.time()-start_time:.2f} seconds\n'
        )

    output.update(
        sam_image=cv2.resize(generated_sam, (1024, 1024)),
        overlap_image=cv2.resize(best_mask_annotated, (1024, 1024)),
        show_overlap_image=cv2.resize(show_overlap, (1024, 1024)),
        final_output=cv2.resize(final_output, (1024, 1024)),
        summary=summary,
        final_output_accordion=gr.Accordion(open=True),
        output=output_text_stream.strip()
    )

    yield tuple(output.values())
