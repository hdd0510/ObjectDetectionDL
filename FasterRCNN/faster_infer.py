import torch
import torchvision.transforms as T
from PIL import Image
import sys
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler

def load_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 3 # 2 class (license_plate, lights) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader, device= ('cuda' if torch.cuda.is_available() else 'cpu'), detection_threshold= 0.5):
    results = []
    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Retrieve the scores and convert to numpy
                scores = output['scores'].data.cpu().numpy()
                class_ids = output['labels']
                # Retrieve the boxes and convert to numpy
                boxes = output['boxes'].data.cpu().numpy()
                
                # Apply the detection threshold
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]
                
                # Convert boxes from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                
                # Collect results in COCO format
                for class_id, box, score in(zip(class_ids, boxes, scores)):
                    result = {
                        "image_id": image_ids[i],
                        "category_id": class_id.item(),
                        "bbox": box.tolist(),  # Convert numpy array to list
                        "score": float(score)  # Convert numpy float to Python float
                    }
                    results.append(result)
    return results

