"""
experiment2.py
This script is used to generate masks using the Florence model and save them to the disk.
All generated masks are concatenated and saved in different formats for further analysis.

Usage:
- Place the input images in the specified folder `images_path`.
- Place the ground truth masks in the specified `mask_path`.

Steps:
1. Load the Florence model.
2. Iterate over each image in the `images_path`.
3. Generate masks using the Florence model.
4. Save the generated masks in different formats.
5. Calculate and save the IoU scores.

Outputs:
- Generated masks in `florence_masks/overlay`.
- Raw masks in `florence_masks/raw`.
- IoU scores in `florence_masks/iou_scores_binary.csv` and `florence_masks/iou_scores_micro.csv`.

Dependencies:
- copy
- os
- pathlib
- numpy
- pandas
- PIL
- sklearn
- supervision
- transformers

"""

import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import jaccard_score

import supervision as sv

from transformers import AutoModelForCausalLM, AutoProcessor


IMAGES_PATH = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/images')
MASK_PATH = Path('/mnt/e/Datasets/kvasir-seg/Kvasir-SEG/masks')


def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


if __name__ == '__main__':
    model_id = 'microsoft/Florence-2-base'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    Path('florence_masks').mkdir(exist_ok=True)
    Path('florence_masks', 'overlay').mkdir(exist_ok=True)
    Path('florence_masks', 'raw').mkdir(exist_ok=True)

    images_path = IMAGES_PATH
    mask_path = MASK_PATH

    iou_scores_binary = []
    iou_scores_micro = []
    for image_name in os.listdir(images_path):
        image = Image.open(images_path / image_name)
        truth_mask = Image.open(mask_path / image_name)
        truth_mask = truth_mask.convert('L')
        truth_mask = np.array(truth_mask) > 0.5

        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        text_caption = 'polyp'
        results = run_example(task_prompt, text_input=text_caption)
        output_image = copy.deepcopy(image)
        # output_segmentation_image = draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)

        # vertices = results['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons'][0]
        polygons = []
        for vertices in results['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons'][0]:
            polygon = [(vertices[i], vertices[i + 1]) for i in range(0, len(vertices), 2)]
            polygons.append(polygon)

        florence_masks = []
        for polygon in polygons:
            florence_mask = Image.new('1', (image.width, image.height), 0)
            ImageDraw.Draw(florence_mask).polygon(polygon, outline=1, fill=1)
            florence_mask = np.array(florence_mask)
            florence_masks.append(florence_mask)

        florence_masks = np.stack(florence_masks, axis=0)
        detections = sv.Detections(np.array([[0, 0, 0, 0]] * len(florence_masks)), np.array(florence_masks),
                                   class_id=np.ones(len(florence_masks), dtype=int))

        florence_mask = sv.MaskAnnotator().annotate(np.array(image), detections)
        Image.fromarray(florence_mask).save(Path('florence_masks', 'overlay', image_name))
        truth_mask = np.array(truth_mask)

        # Merge florence_masks
        florence_masks = np.sum(florence_masks, axis=0, dtype=bool)
        iou_micro = jaccard_score(truth_mask, florence_masks, average='micro')
        iou_scores_micro.append(iou_micro)
        iou_binary = jaccard_score(truth_mask.ravel(), florence_masks.ravel(), average='binary')
        iou_scores_binary.append(iou_binary)

        print(f'IoU Binary: {iou_binary}, IoU Micro: {iou_micro}')
        Image.fromarray(florence_masks).save(Path('florence_masks', 'raw', image_name))

    df_iou_scores_binary = pd.DataFrame(iou_scores_binary)
    df_iou_scores_binary.index = os.listdir(images_path)
    df_iou_scores_binary.to_csv(Path('florence_masks', 'iou_scores_binary.csv'), index=False)

    df_iou_scores_micro = pd.DataFrame(iou_scores_micro)
    df_iou_scores_micro.index = os.listdir(images_path)
    df_iou_scores_micro.to_csv(Path('florence_masks', 'iou_scores_micro.csv'), index=False)

    print(f'Average IoU Binary: {np.mean(iou_scores_binary)}, Average IoU Micro: {np.mean(iou_scores_micro)}')
