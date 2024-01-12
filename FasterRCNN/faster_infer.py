import torch
import torchvision.transforms as T
from PIL import Image
import sys
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)

    num_classes = 3 # 2 class (license_plate, lights) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_checkpoint = torch.load(model_path)
    model.load_state_dict(model_checkpoint)
    model.eval()
    return model

def prediction(model, directory, device= ('cuda' if torch.cuda.is_available() else 'cpu'), detection_threshold= 0.5):
    results = []
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    
    # Loop over all files in the directory
    for i, image_file in enumerate(os.listdir(directory)):
        # Make sure to read only image files (you may need to adjust the extension list)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        # Construct the full path to the image file
        image_path = os.path.join(directory, image_file)
        # Open the image file with OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a tensor that the model can understand
        val_transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                    ], bbox_params=A.BboxParams(format='pascal_voc')
                )
        image_tensor = val_transform(image=image)['image']
        
        with torch.no_grad():
            # Get the model output
            output = model(image_tensor)
        
        # Retrieve scores, labels, and boxes from the model's output
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()
        boxes = output[0]['boxes'].data.cpu().numpy()

        # Apply the detection threshold
        indices = scores >= detection_threshold
        boxes = boxes[indices].astype(np.int32)
        scores = scores[indices]
        labels = labels[indices]

        # Convert boxes from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Collect results
        for label, box, score in zip(labels, boxes, scores):
            result = {
                'filename': image_file,
                "image_id": i+1,
                "category_id": int(label),  # Convert numpy int to Python int
                "bbox": box.tolist(),  # Convert numpy array to list
                "score": float(score)  # Convert numpy float to Python float
            }
            results.append(result)
    
    return results
def create_coco_json(image_dir, gt_dir, category_id_mapping, img_width, img_height):
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories to COCO JSON
    for category_name, category_id in category_id_mapping.items():
        coco_format['categories'].append({
            'id': category_id,
            'name': category_name,
            'supercategory': category_name
        })

    # Populate images and annotations
    annotation_id = 1
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = i+1
            # Add image information
            coco_format['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': img_width,
                'height': img_height
            })

            # Corresponding ground truth file
            gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            if os.path.exists(gt_filepath):
                with open(gt_filepath, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, bbox_width, bbox_height = [
                            float(x) for x in line.strip().split()
                        ]

                        # Convert to COCO format
                        x_min = (x_center - (bbox_width / 2)) * img_width
                        y_min = (y_center - (bbox_height / 2)) * img_height
                        width = bbox_width * img_width
                        height = bbox_height * img_height

                        # Add annotation information
                        coco_format['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': int(class_id)+1,
                            'bbox': [x_min, y_min, width, height],
                            'area': width * height,
                            'iscrowd': 0,
                            # 'segmentation': [[0 for i in range(640)] for i in range(640)]
                        })
                        annotation_id += 1
    return coco_format

def main(model_path, data_path):
    model = load_model(model_path)
    gt_dir = f'{data_path}/labels'
    image_dir = f'{data_path}/images'
    output_json_path = 'ground_truths.json'

    tide_predictions = prediction(model, image_dir)

    # Initialize the JSON structure
    category_id_mapping = {'license plate': 2, 'lights': 1}
    img_width = 640
    img_height = 640

    coco_format = create_coco_json(image_dir, gt_dir, category_id_mapping, img_width, img_height)
    # Write the COCO format JSON to a file
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    with open('predictions.json', 'w') as f:
        json.dump(tide_predictions, f)
    
    coco_gt = COCO('/kaggle/working/ground_truths.json')  # path to the JSON with ground truth annotations
    coco_dt = coco_gt.loadRes('/kaggle/working/predictions.json')  # path to the JSON with detection results
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', help='Path to the YOLOv8 model file', required=True)
    args = parser.parse_args()

    main(args.model_path, args.data_path)
