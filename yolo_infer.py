import torch
import yaml
import cv2
import argparse
import os 
import json
from yolov8 import visual_prediction, create_coco_json, get_yolo_predictions  # This assumes you have a models file with your YOLOv8 model definition
import random
from tidecv import TIDE
import tidecv.datasets as datasets
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

def load_yaml_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path):
    model = YOLO(model_path)  
    return model

def main(test_path, model_path, yaml_path):
    data = load_yaml_config(yaml_path)
    data['val'] = test_path
    with open(os.path.join('/kaggle/working/', 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    model = load_model(model_path)
    model.val()

    image_dir = f'{test_path}/images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    # Choose a random image file
    random_image_file = random.choice(image_files)

    # Load the image using OpenCV
    image = cv2.imread(random_image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    for result in results:
        visual_prediction(result)

    gt_dir = '/kaggle/input/licenseplate/test/labels'
    image_dir = '/kaggle/input/licenseplate/test/images'
    output_json_path = 'ground_truths.json'

    tide_predictions = get_yolo_predictions(model, image_dir)

    # Initialize the JSON structure
    category_id_mapping = {'license plate': 1, 'lights': 0}
    img_width = 640
    img_height = 640

    coco_format = create_coco_json(image_dir, gt_dir, category_id_mapping, img_width, img_height)
    # Write the COCO format JSON to a file
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    with open('predictions.json', 'w') as f:
        json.dump(tide_predictions, f)
    

    pred = datasets.COCOResult('/kaggle/working/predictions.json')
    gt = datasets.COCO('/kaggle/working/ground_truths.json')
    
    coco_gt = COCO('/kaggle/working/ground_truths.json')  # path to the JSON with ground truth annotations
    coco_dt = coco_gt.loadRes('/kaggle/working/predictions.json')  # path to the JSON with detection results
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    tide = TIDE()
    tide.evaluate_range(gt, pred, mode=TIDE.BOX)

    tide.summarize()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv8 model file', required=True)
    parser.add_argument('--yaml_path', type=str, help='Path to the YOLOv8 data.yaml file', required=True)
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.yaml_path)
