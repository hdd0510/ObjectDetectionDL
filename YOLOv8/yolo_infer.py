import torch
import yaml
import cv2
import argparse
import os 
import json
import random
import shutil
from tidecv import TIDE
import tidecv.datasets as datasets
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import matplotlib.pyplot as plt

def visual_prediction(image, result):
    boxes = result.boxes.cpu().numpy()
    xyxys = boxes.xyxy
    class_ids = boxes.cls
    confidences = boxes.conf
    class_labels = {
        0: 'light',
        1: 'license plate'
    }
    for box, class_id, confidence in zip(xyxys, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_labels[class_id]
        label = f'{class_name} {confidence:.2f}'
        
        # Tính kích thước của nhãn để tạo background
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 0, 0), thickness=cv2.FILLED)
        
        # Vẽ nhãn lên ảnh
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vẽ bounding box xung quanh đối tượng
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Sử dụng Matplotlib để hiển thị ảnh
    plt.figure(figsize=(12, 8))  # Có thể điều chỉnh kích thước nếu muốn
    plt.imshow(image)
    plt.axis('off')  # Ẩn trục tọa độ
    plt.show()
def get_yolo_predictions(model, image_dir):
    predictions = []
    for i, filename in enumerate(os.listdir(image_dir)):
        # Get the image ID from the mapping
        image_id = i

        # Construct the full path to the image
        image_path = os.path.join(image_dir, filename)

        # Perform inference
        results = model(image_path, verbose = False)

        for result in results:
            for box in result.boxes:
                xyxys = box.xyxy
                for coordinates in xyxys:
                    x1, y1, x2, y2 = coordinates  # Modify as necessary to match the box attribute
                    score = box.conf  # Confidence score
                    class_id = box.cls  # Class ID
                    image_id = i  # If you have unique image IDs, use those instead

                    # Construct the prediction dictionary for TIDE
                    tide_pred = {
                        'image_id': image_id,  # You may need to map this to actual image IDs if necessary
                        'category_id': class_id.item(),  # Map this to COCO class IDs if necessary
                        'bbox': [x1.item(), y1.item(), (x2-x1).item(), (y2-y1).item()],  # Convert to [x_min, y_min, width, height]
                        'score': score.item(),
                        'filename': filename
                    }
                    predictions.append(tide_pred)
    return predictions
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
            image_id = i
            image_path = os.path.join(image_dir, filename)
            
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
                            'category_id': int(class_id),
                            'bbox': [x_min, y_min, width, height],
                            'area': width * height,
                            'iscrowd': 0,
                            'segmentation': [[0 for i in range(640)] for i in range(640)]
                        })
                        annotation_id += 1
    return coco_format
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
    images = cv2.imread(random_image_file)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    results = model(images)
    for result in results:
        visual_prediction(images, result)

    # gt_dir = '/kaggle/input/licenseplate/test/labels'
    # image_dir = '/kaggle/input/licenseplate/test/images'
    # output_json_path = 'ground_truths.json'

    # tide_predictions = get_yolo_predictions(model, image_dir)

    # # Initialize the JSON structure
    # category_id_mapping = {'license plate': 1, 'lights': 0}
    # img_width = 640
    # img_height = 640

    # coco_format = create_coco_json(image_dir, gt_dir, category_id_mapping, img_width, img_height)
    # # Write the COCO format JSON to a file
    # with open(output_json_path, 'w') as f:
    #     json.dump(coco_format, f, indent=2)
    # with open('predictions.json', 'w') as f:
    #     json.dump(tide_predictions, f)
    

    # pred = datasets.COCOResult('/kaggle/working/predictions.json')
    # gt = datasets.COCO('/kaggle/working/ground_truths.json')
    
    # coco_gt = COCO('/kaggle/working/ground_truths.json')  # path to the JSON with ground truth annotations
    # coco_dt = coco_gt.loadRes('/kaggle/working/predictions.json')  # path to the JSON with detection results
    # coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # # Run evaluation
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    # tide = TIDE()
    # tide.evaluate_range(gt, pred, mode=TIDE.BOX)
    

    # tide.summarize()

    class_labels = {
        0: 'light',
        1: 'license plate'
    }
    def save_predictions_yolov8(model, image_dir, save_dir, detection_threshold=0.5, device='cuda'):
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load YOLOv8 model
        # Assume 'model' is a YOLOv8 object that has a method 'predict' which takes a directory path as input
        
        # Evaluate the model on the given image directory
        results = model.predict(image_dir, conf=detection_threshold, device=device)
        
        # Assume 'results' is a list of predictions where each prediction contains:
        # image path, boxes, scores, and labels
        
        for result in results:
            image_path = result.path
            boxes = result.boxes
            scores = boxes.conf
            labels = boxes.cls
            
            # Load image
            image = cv2.imread(image_path)
            for box, score, label in zip(boxes.xyxy, scores, labels):
                x_min, y_min, x_max, y_max = map(int,box.data.cpu().numpy())
                # Draw rectangle on the image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 220), 2)
                
                # Add label and score to the image
                label_str = f'{class_labels[label.item()]}: {score:.2f}'
                cv2.putText(image, label_str, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 2)
            
            # Save image with predictions
            basename = os.path.basename(image_path)
            save_path = os.path.join(save_dir, f'predicted_{basename}')
            cv2.imwrite(save_path, image)

    # Call the function with YOLOv8 model, image directory, and path to save directory
    save_predictions_yolov8(model, f'{test_path}/images', 'prediction')
    shutil.make_archive('/kaggle/working/predictions', 'zip', '/kaggle/working/predictions')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv8 model file', required=True)
    parser.add_argument('--yaml_path', type=str, help='Path to the YOLOv8 data.yaml file', required=True)
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.yaml_path)
