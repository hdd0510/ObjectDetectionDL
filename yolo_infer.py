import torch
import yaml
import cv2
import argparse
# from models import *  # This assumes you have a models.py file with your YOLOv8 model definition
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
    model = load_model(model_path)
    data['val'] = test_path
    model.val()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv8 model file', required=True)
    parser.add_argument('--yaml_path', type=str, help='Path to the YOLOv8 data.yaml file', required=True)
    args = parser.parse_args()

    main(args.image_path, args.model_path, args.yaml_path)
