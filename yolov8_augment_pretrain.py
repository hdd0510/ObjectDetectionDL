{"metadata":{"kaggle":{"accelerator":"gpu","dataSources":[{"sourceId":7020108,"sourceType":"datasetVersion","datasetId":4036736},{"sourceId":7156048,"sourceType":"datasetVersion","datasetId":4132641},{"sourceId":7334480,"sourceType":"datasetVersion","datasetId":4257845},{"sourceId":7339277,"sourceType":"datasetVersion","datasetId":4261092},{"sourceId":7342981,"sourceType":"datasetVersion","datasetId":4263626}],"dockerImageVersionId":30627,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":true},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"codemirror_mode":{"name":"ipython","version":3},"file_extension":".py","mimetype":"text/x-python","name":"python","nbconvert_exporter":"python","pygments_lexer":"ipython3","version":"3.10.12"},"papermill":{"default_parameters":{},"duration":10887.177824,"end_time":"2024-01-04T11:04:30.486019","environment_variables":{},"exception":null,"input_path":"__notebook__.ipynb","output_path":"__notebook__.ipynb","parameters":{},"start_time":"2024-01-04T08:03:03.308195","version":"2.4.0"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# !pip install albumentations","metadata":{"papermill":{"duration":0.01529,"end_time":"2024-01-04T08:03:06.664575","exception":false,"start_time":"2024-01-04T08:03:06.649285","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:07.566111Z","iopub.execute_input":"2024-01-06T18:14:07.566959Z","iopub.status.idle":"2024-01-06T18:14:07.570779Z","shell.execute_reply.started":"2024-01-06T18:14:07.566925Z","shell.execute_reply":"2024-01-06T18:14:07.569742Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import os\nimport torch\nimport numpy as np\nfrom torch.utils.data import Dataset, DataLoader\nfrom PIL import Image\nimport cv2\nimport torch.optim as optim\nimport torch.nn as nn \nimport albumentations as A\nfrom albumentations.pytorch import ToTensorV2\nimport os\nimport yaml\nfrom torchvision.transforms import ToPILImage, transforms\nimport torch\n\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\n","metadata":{"_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","papermill":{"duration":5.714491,"end_time":"2024-01-04T08:03:12.386216","exception":false,"start_time":"2024-01-04T08:03:06.671725","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:07.574533Z","iopub.execute_input":"2024-01-06T18:14:07.575358Z","iopub.status.idle":"2024-01-06T18:14:12.471728Z","shell.execute_reply.started":"2024-01-06T18:14:07.575322Z","shell.execute_reply":"2024-01-06T18:14:12.470917Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"image_path = []\nTRAIN_IMG_DIR = '/kaggle/input/licenseplate/train/images'\nfor root, dirs, files in os.walk(TRAIN_IMG_DIR):\n    for file in files:\n        # create path\n        path = os.path.join(root,file)\n        # add path to list\n        image_path.append(path)\nlen(image_path)","metadata":{"papermill":{"duration":2.384875,"end_time":"2024-01-04T08:03:14.778219","exception":false,"start_time":"2024-01-04T08:03:12.393344","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:12.473302Z","iopub.execute_input":"2024-01-06T18:14:12.473684Z","iopub.status.idle":"2024-01-06T18:14:15.093224Z","shell.execute_reply.started":"2024-01-06T18:14:12.473647Z","shell.execute_reply":"2024-01-06T18:14:15.092307Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"label_path = []\nTRAIN_LABEL_DIR = '/kaggle/input/licenseplate/train/labels'\nfor root, dirs, files in os.walk(TRAIN_LABEL_DIR):\n    for file in files:\n        # create path\n        path = os.path.join(root,file)\n        # add path to list\n        label_path.append(path)\nlen(label_path)","metadata":{"papermill":{"duration":3.268192,"end_time":"2024-01-04T08:03:18.053378","exception":false,"start_time":"2024-01-04T08:03:14.785186","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:15.094613Z","iopub.execute_input":"2024-01-06T18:14:15.095045Z","iopub.status.idle":"2024-01-06T18:14:17.838368Z","shell.execute_reply.started":"2024-01-06T18:14:15.095007Z","shell.execute_reply":"2024-01-06T18:14:17.837364Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"class LicensePlateDataset(Dataset):\n    def __init__(self, image_dir, label_dir, transform=None):\n        self.image_dir = image_dir\n        self.label_dir = label_dir\n        self.transform = transform\n        self.images = os.listdir(image_dir)\n\n    def __len__(self):\n        return len(self.images)\n\n    def __getitem__(self, index):\n        image_path = os.path.join(self.image_dir, self.images[index])\n        label_path = os.path.join(self.label_dir, self.images[index].replace('jpg', 'txt').replace('txt', 'jpg', 1))\n        \n        image = cv2.imread(image_path) # read in BGR format\n        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n        boxes = []\n        labels = []\n        \n        with open(label_path, 'r') as f:\n            for line in f.readlines():\n                class_label, x_center, y_center, width, height = [\n                    float(x) if float(x) != int(float(x)) else int(x) \n                    for x in line.replace('\\n', '').split()\n                ]\n                boxes.append([x_center, y_center, width, height])\n                labels.append(class_label)\n        \n        # Convert boxes and labels to tensors\n        boxes = torch.tensor(boxes, dtype=torch.float32)\n        labels = torch.tensor(labels, dtype=torch.int64)\n        \n        target = {}\n        target[\"boxes\"] = boxes\n        target[\"labels\"] = labels\n        \n        if self.transform:\n            transformed = self.transform(image=image, bboxes=boxes, labels=labels)\n            image = transformed['image']\n            target = {\n                'boxes': torch.tensor(transformed['bboxes']),\n                'labels': torch.tensor(transformed['labels'])\n            }\n        return image, target\n","metadata":{"papermill":{"duration":0.021859,"end_time":"2024-01-04T08:03:18.082226","exception":false,"start_time":"2024-01-04T08:03:18.060367","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.840584Z","iopub.execute_input":"2024-01-06T18:14:17.840879Z","iopub.status.idle":"2024-01-06T18:14:17.852872Z","shell.execute_reply.started":"2024-01-06T18:14:17.840853Z","shell.execute_reply":"2024-01-06T18:14:17.852024Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"transform = A.Compose([\n    A.HorizontalFlip(p=0.5),\n    A.RandomBrightnessContrast(p=0.2),\n    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),\n    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(1, 1), p=0.5),\n    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),\n    A.GaussianBlur(blur_limit=(3, 7), p=0.2),\n    ToTensorV2()\n], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))","metadata":{"papermill":{"duration":0.015951,"end_time":"2024-01-04T08:03:18.105153","exception":false,"start_time":"2024-01-04T08:03:18.089202","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.854017Z","iopub.execute_input":"2024-01-06T18:14:17.854281Z","iopub.status.idle":"2024-01-06T18:14:17.870463Z","shell.execute_reply.started":"2024-01-06T18:14:17.854256Z","shell.execute_reply":"2024-01-06T18:14:17.869659Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"transform = transform\ntrain_dataset = LicensePlateDataset(image_dir= TRAIN_IMG_DIR, label_dir= TRAIN_LABEL_DIR, transform= transform)","metadata":{"papermill":{"duration":0.021536,"end_time":"2024-01-04T08:03:18.133519","exception":false,"start_time":"2024-01-04T08:03:18.111983","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.871620Z","iopub.execute_input":"2024-01-06T18:14:17.871964Z","iopub.status.idle":"2024-01-06T18:14:17.888826Z","shell.execute_reply.started":"2024-01-06T18:14:17.871926Z","shell.execute_reply":"2024-01-06T18:14:17.888073Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"# Visualize","metadata":{"papermill":{"duration":0.006448,"end_time":"2024-01-04T08:03:18.146673","exception":false,"start_time":"2024-01-04T08:03:18.140225","status":"completed"},"tags":[]}},{"cell_type":"code","source":"class_labels = {\n    0: 'light',\n    1: 'license plate'\n}\n\ndef visualize_image_with_bbox(image, target, class_mark = True):\n    image = np.transpose(image.numpy(), (1, 2, 0))  # Convert from CHW to HWC format for matplotlib\n    bboxes = target['boxes']\n    print()\n    h, w, _ = image.shape  # Get image dimensions\n    \n    fig, ax = plt.subplots(1, figsize=(12, 7))\n    ax.imshow(image)\n    \n    \n    for i, bbox in enumerate(bboxes):\n        x_center, y_center, bbox_width, bbox_height = bbox\n        x_min = (x_center - bbox_width / 2) * w\n        y_min = (y_center - bbox_height / 2) * h\n        box_width = bbox_width * w\n        box_height = bbox_height * h\n\n        rect = patches.Rectangle(\n            (x_min, y_min),\n            box_width,\n            box_height,\n            linewidth=2,\n            edgecolor='r',\n            facecolor='none'\n        )\n        if class_mark:\n            class_id = target['labels'][i].item()\n            label_text = class_labels[class_id]\n            ax.text(x_min, y_min, label_text, color='white', \n                    verticalalignment='top', bbox={'color': 'red', 'pad': 0})\n        ax.add_patch(rect)\n    \n    plt.axis('off')\n    plt.show()","metadata":{"papermill":{"duration":0.018464,"end_time":"2024-01-04T08:03:18.171735","exception":false,"start_time":"2024-01-04T08:03:18.153271","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.890018Z","iopub.execute_input":"2024-01-06T18:14:17.890280Z","iopub.status.idle":"2024-01-06T18:14:17.899291Z","shell.execute_reply.started":"2024-01-06T18:14:17.890257Z","shell.execute_reply":"2024-01-06T18:14:17.898396Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"","metadata":{"papermill":{"duration":0.006441,"end_time":"2024-01-04T08:03:18.185001","exception":false,"start_time":"2024-01-04T08:03:18.178560","status":"completed"},"tags":[]},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# for img, label in train_dataset:\n#     # Kiểm tra nếu nhãn là một tensor chứa nhiều nhãn\n#     if not (label['labels'] == 1).all():\n#         print('Sample with label not equal to 1 found')\n#         visualize_image_with_bbox(img, label)","metadata":{"papermill":{"duration":0.01413,"end_time":"2024-01-04T08:03:18.205800","exception":false,"start_time":"2024-01-04T08:03:18.191670","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.900271Z","iopub.execute_input":"2024-01-06T18:14:17.900546Z","iopub.status.idle":"2024-01-06T18:14:17.912332Z","shell.execute_reply.started":"2024-01-06T18:14:17.900522Z","shell.execute_reply":"2024-01-06T18:14:17.911526Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport numpy as np\nimport random\nimport albumentations as A\nfrom albumentations.pytorch.transforms import ToTensorV2\n\ndef visualize_sample(dataset, class_label = True):\n    # Get a random sample\n    idx = random.randint(0, len(dataset) - 1)\n    image, target = dataset[idx]\n# Visualization function\n    image = np.transpose(image.numpy(), (1, 2, 0))  # Change from CHW to HWC format for matplotlib\n    h, w, _ = image.shape  # Get image dimensions\n    print(h)\n    bboxes = target['boxes']  # Assuming YOLO format (x_center, y_center, width, height)\n    # Create the plot\n    fig, ax = plt.subplots(1, figsize=(12, 7))\n    ax.imshow(image)\n    \n    # Add the bounding boxes\n    for bbox in bboxes:\n        x_center, y_center, bbox_width, bbox_height = bbox\n        x_min = (x_center - bbox_width / 2) * w\n        y_min = (y_center - bbox_height / 2) * h\n        box_width = bbox_width * w\n        box_height = bbox_height * h\n        \n        rect = patches.Rectangle(\n            (x_min, y_min), \n            box_width, \n            box_height, \n            linewidth=2, \n            edgecolor='r', \n            facecolor='none'\n        )\n        ax.add_patch(rect)\n        if class_label:\n            label_text = 'license plate'\n            ax.text(x_min, y_min, label_text, color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})\n#         plt.axis('off')  # Hide the axes\n    plt.show()\n\n# Visualize a random sample\nvisualize_sample(train_dataset)","metadata":{"papermill":{"duration":0.462363,"end_time":"2024-01-04T08:03:18.675680","exception":false,"start_time":"2024-01-04T08:03:18.213317","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:17.913434Z","iopub.execute_input":"2024-01-06T18:14:17.913797Z","iopub.status.idle":"2024-01-06T18:14:18.553388Z","shell.execute_reply.started":"2024-01-06T18:14:17.913764Z","shell.execute_reply":"2024-01-06T18:14:18.552470Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"os.makedirs('train/images', exist_ok=True)\nos.makedirs('train/labels', exist_ok=True)","metadata":{"papermill":{"duration":0.018284,"end_time":"2024-01-04T08:03:18.704279","exception":false,"start_time":"2024-01-04T08:03:18.685995","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:18.556813Z","iopub.execute_input":"2024-01-06T18:14:18.557103Z","iopub.status.idle":"2024-01-06T18:14:18.561892Z","shell.execute_reply.started":"2024-01-06T18:14:18.557077Z","shell.execute_reply":"2024-01-06T18:14:18.560946Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"image_paths = []\nlabel_paths = []\n\nfor i, (image, target) in enumerate(train_dataset):\n    # Lưu hình ảnh\n    pil_image = ToPILImage()(image.squeeze(0))  # Chuyển tensor sang PIL Image\n    image_path = f'train/images/image_id{i}.jpg'\n    pil_image.save(image_path)\n    image_paths.append(image_path)\n\n    # Lưu label\n    label_path = f'train/labels/image_id{i}.txt'\n    with open(label_path, 'w') as file:\n        for box, label in zip(target['boxes'], target['labels']):\n            # Giả sử bounding box đã ở định dạng x_center, y_center, width, height\n            file.write(f'{label.item()} {box[0].item()} {box[1].item()} {box[2].item()} {box[3].item()}\\n')\n    label_paths.append(label_path)\nprint('The new directories are saved')","metadata":{"papermill":{"duration":124.554375,"end_time":"2024-01-04T08:05:23.268913","exception":false,"start_time":"2024-01-04T08:03:18.714538","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:18.563005Z","iopub.execute_input":"2024-01-06T18:14:18.563266Z","iopub.status.idle":"2024-01-06T18:14:21.051400Z","shell.execute_reply.started":"2024-01-06T18:14:18.563243Z","shell.execute_reply":"2024-01-06T18:14:21.049229Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import yaml\nimport os\n\ndata_yaml_path = '/kaggle/input/licenseplate/data.yaml'\n\nwith open(data_yaml_path, 'r') as file:\n    data = yaml.safe_load(file)\n\nprint(data)\n\ndata['train'] = '/kaggle/working/train'\ndata['val'] = '/kaggle/input/licenseplate/valid'\ndata['test'] = '/kaggle/input/licenseplate/test'\n\nwith open(os.path.join('/kaggle/working/', 'data.yaml'), 'w') as f:\n    yaml.dump(data, f, default_flow_style=False)\n\ndata_yaml_path = '/kaggle/working/data.yaml'\n    \nwith open(data_yaml_path, 'w') as file:\n    yaml.dump(data, file, default_flow_style=False)\n\nwith open(data_yaml_path, 'r') as file:\n    updated_data = yaml.safe_load(file)\nprint(updated_data)\n","metadata":{"papermill":{"duration":0.03429,"end_time":"2024-01-04T08:05:23.314875","exception":false,"start_time":"2024-01-04T08:05:23.280585","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.052348Z","iopub.status.idle":"2024-01-06T18:14:21.052722Z","shell.execute_reply.started":"2024-01-06T18:14:21.052531Z","shell.execute_reply":"2024-01-06T18:14:21.052547Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"!pip install wandb\n!wandb login 'af3fff39f107c47a5441bad9ba81d9c46a34914b'","metadata":{"papermill":{"duration":4.053414,"end_time":"2024-01-04T08:05:27.380094","exception":false,"start_time":"2024-01-04T08:05:23.326680","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.054256Z","iopub.status.idle":"2024-01-06T18:14:21.054599Z","shell.execute_reply.started":"2024-01-06T18:14:21.054433Z","shell.execute_reply":"2024-01-06T18:14:21.054450Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"!pip install ultralytics","metadata":{"papermill":{"duration":14.335222,"end_time":"2024-01-04T08:05:41.725894","exception":false,"start_time":"2024-01-04T08:05:27.390672","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.056393Z","iopub.status.idle":"2024-01-06T18:14:21.057110Z","shell.execute_reply.started":"2024-01-06T18:14:21.056927Z","shell.execute_reply":"2024-01-06T18:14:21.056949Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"from ultralytics import YOLO\n \n# Load the model.\nmodel = YOLO('yolov8n.pt')\n# Training.\nresults = model.train(\n   data='data.yaml',\n   imgsz=640,\n   epochs=150,\n   batch=8,\n   name='yolov8n_augmented_pretrained'\n)","metadata":{"papermill":{"duration":10544.405009,"end_time":"2024-01-04T11:01:26.143097","exception":false,"start_time":"2024-01-04T08:05:41.738088","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.058357Z","iopub.status.idle":"2024-01-06T18:14:21.058971Z","shell.execute_reply.started":"2024-01-06T18:14:21.058790Z","shell.execute_reply":"2024-01-06T18:14:21.058808Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"","metadata":{"papermill":{"duration":13.820139,"end_time":"2024-01-04T11:01:53.664815","exception":false,"start_time":"2024-01-04T11:01:39.844676","status":"completed"},"tags":[]},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"","metadata":{"papermill":{"duration":14.481036,"end_time":"2024-01-04T11:02:21.805768","exception":false,"start_time":"2024-01-04T11:02:07.324732","status":"completed"},"tags":[]},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"model = YOLO('/kaggle/working/runs/detect/yolov8n_custom/weights/best.pt')  # load a custom model\n\n# Validate the model\nmetrics = model.val()  # no arguments needed, dataset and settings remembered\nmetrics.box.map    # map50-95\nmetrics.box.map50  # map50\nmetrics.box.map75  # map75\nmetrics.box.maps  ","metadata":{"papermill":{"duration":38.277227,"end_time":"2024-01-04T11:03:15.269312","exception":false,"start_time":"2024-01-04T11:02:36.992085","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.060216Z","iopub.status.idle":"2024-01-06T18:14:21.060555Z","shell.execute_reply.started":"2024-01-06T18:14:21.060391Z","shell.execute_reply":"2024-01-06T18:14:21.060408Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# có augment, không pretrained","metadata":{"papermill":{"duration":13.731573,"end_time":"2024-01-04T11:03:42.924151","exception":false,"start_time":"2024-01-04T11:03:29.192578","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-01-06T18:14:21.062333Z","iopub.status.idle":"2024-01-06T18:14:21.063016Z","shell.execute_reply.started":"2024-01-06T18:14:21.062778Z","shell.execute_reply":"2024-01-06T18:14:21.062805Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"def visual_prediction(result):\n    boxes = result.boxes.cpu().numpy()\n    xyxys = boxes.xyxy\n    class_ids = boxes.cls\n    confidences = boxes.conf\n    \n    for box, class_id, confidence in zip(xyxys, class_ids, confidences):\n        x1, y1, x2, y2 = map(int, box)\n        class_name = class_labels[class_id]\n        label = f'{class_name} {confidence:.2f}'\n        \n        # Tính kích thước của nhãn để tạo background\n        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)\n        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 0, 0), thickness=cv2.FILLED)\n        \n        # Vẽ nhãn lên ảnh\n        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)\n\n        # Vẽ bounding box xung quanh đối tượng\n        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)\n    # Sử dụng Matplotlib để hiển thị ảnh\n    plt.figure(figsize=(12, 8))  # Có thể điều chỉnh kích thước nếu muốn\n    plt.imshow(image)\n    plt.axis('off')  # Ẩn trục tọa độ\n    plt.show()","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.064160Z","iopub.status.idle":"2024-01-06T18:14:21.064670Z","shell.execute_reply.started":"2024-01-06T18:14:21.064499Z","shell.execute_reply":"2024-01-06T18:14:21.064516Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"image = cv2.imread('/kaggle/input/licenseplate/test/images/000188bf-2b4d-4f97-9fe6-9ab8d23c8ae2_jpg.rf.9754c504d16ce84aa0c5754038ae2412.jpg')\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\nresults = model(image)\nfor result in results:\n    visual_prediction(result)","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.065998Z","iopub.status.idle":"2024-01-06T18:14:21.066337Z","shell.execute_reply.started":"2024-01-06T18:14:21.066173Z","shell.execute_reply":"2024-01-06T18:14:21.066189Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"tide_predictions = []\n\n# Assuming 'results' is your YOLO model's output object\nfor i, result in enumerate(results):\n    # Convert boxes to TIDE format\n    for box in result.boxes:\n        xyxys = box.xyxy\n        for coordinates in xyxys:\n            x1, y1, x2, y2 = coordinates  # Modify as necessary to match the box attribute\n            score = box.conf  # Confidence score\n            class_id = box.cls  # Class ID\n            image_id = i  # If you have unique image IDs, use those instead\n\n            # Construct the prediction dictionary for TIDE\n            tide_pred = {\n                'image_id': image_id,  # You may need to map this to actual image IDs if necessary\n                'category_id': class_id,  # Map this to COCO class IDs if necessary\n                'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to [x_min, y_min, width, height]\n                'score': score\n            }\n            tide_predictions.append(tide_pred)\nprint(tide_predictions[0])","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.068079Z","iopub.status.idle":"2024-01-06T18:14:21.068811Z","shell.execute_reply.started":"2024-01-06T18:14:21.068598Z","shell.execute_reply":"2024-01-06T18:14:21.068615Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"tide_ground_truths = []\n\n# List all text files in the directory\ndirectory = '/kaggle/input/licenseplate/test/labels'\ncnt = 0\nfor i, filename in enumerate(os.listdir(directory)):\n    cnt += 1\n    if filename.endswith('.txt'):\n        filepath = os.path.join(directory, filename)\n        image_id = i\n        # Read annotation lines from the file\n        with open(filepath, 'r') as file:\n            for line in file:\n                class_id, x_center, y_center, width, height = map(float, line.split())\n\n                # Convert from normalized [0, 1] range to absolute coordinates\n                # Assuming the image dimensions are known and stored in `image_width` and `image_height`\n                x_min = (x_center - width / 2) * 640\n                y_min = (y_center - height / 2) * 640\n                abs_width = width * 640\n                abs_height = height * 640\n\n                # Create the ground truth dictionary for TIDE\n                ground_truth = {\n                    'image_id': image_id,\n                    'category_id': int(class_id),\n                    'bbox': [x_min, y_min, abs_width, abs_height]\n                }\n\n                tide_ground_truths.append(ground_truth)\nprint(len(tide_ground_truths))","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.069829Z","iopub.status.idle":"2024-01-06T18:14:21.070180Z","shell.execute_reply.started":"2024-01-06T18:14:21.070015Z","shell.execute_reply":"2024-01-06T18:14:21.070031Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"def get_yolo_predictions(model, image_dir):\n    predictions = []\n    for i, filename in enumerate(os.listdir(image_dir)):\n        cnt += 1\n        # Get the image ID from the mapping\n        image_id = i\n\n        # Construct the full path to the image\n        image_path = os.path.join(image_dir, filename)\n\n        # Perform inference\n        results = model(image_path, verbose = False)\n\n        for result in results:\n            for box in result.boxes:\n                xyxys = box.xyxy\n                for coordinates in xyxys:\n                    x1, y1, x2, y2 = coordinates  # Modify as necessary to match the box attribute\n                    score = box.conf  # Confidence score\n                    class_id = box.cls  # Class ID\n                    image_id = i  # If you have unique image IDs, use those instead\n\n                    # Construct the prediction dictionary for TIDE\n                    tide_pred = {\n                        'image_id': image_id,  # You may need to map this to actual image IDs if necessary\n                        'category_id': class_id.item(),  # Map this to COCO class IDs if necessary\n                        'bbox': [x1.item(), y1.item(), (x2-x1).item(), (y2-y1).item()],  # Convert to [x_min, y_min, width, height]\n                        'score': score.item(),\n                        'filename': filename\n                    }\n                    predictions.append(tide_pred)\n    return predictions","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.071890Z","iopub.status.idle":"2024-01-06T18:14:21.072583Z","shell.execute_reply.started":"2024-01-06T18:14:21.072400Z","shell.execute_reply":"2024-01-06T18:14:21.072418Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"tide_predictions = get_yolo_predictions(model,'/kaggle/input/licenseplate/test/images')\nprint(len(tide_predictions))","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.074129Z","iopub.status.idle":"2024-01-06T18:14:21.074839Z","shell.execute_reply.started":"2024-01-06T18:14:21.074560Z","shell.execute_reply":"2024-01-06T18:14:21.074590Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.075951Z","iopub.status.idle":"2024-01-06T18:14:21.076306Z","shell.execute_reply.started":"2024-01-06T18:14:21.076140Z","shell.execute_reply":"2024-01-06T18:14:21.076158Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import json\nimport os\n\n# Replace these with the actual paths and values\ngt_dir = '/kaggle/input/licenseplate/test/labels'\nimage_dir = '/kaggle/input/licenseplate/test/images'\noutput_json_path = 'ground_truths.json'\n\n# Initialize the JSON structure\ncoco_format = {\n    'images': [],\n    'annotations': [],\n    'categories': []\n}\n\n# Let's assume you already have a list of category names and their corresponding IDs\ncategory_id_mapping = {'license plate': 1, 'lights': 0}  # etc.\n\n# Add categories to COCO JSON\nfor category_name, category_id in category_id_mapping.items():\n    coco_format['categories'].append({\n        'id': category_id,\n        'name': category_name,\n        'supercategory': category_name\n    })\n\n# Populate images and annotations\nannotation_id = 1\nfor i, filename in enumerate(os.listdir(image_dir)):\n    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):\n        image_id = i\n        image_path = os.path.join(image_dir, filename)\n        \n        # Add image information\n        coco_format['images'].append({\n            'id': image_id,\n            'file_name': filename,\n            'width': 640,  # Replace with actual image width\n            'height': 640  # Replace with actual image height\n        })\n\n        # Corresponding ground truth file\n        gt_filepath = os.path.join(gt_dir, filename.replace('jpg', 'txt').replace('txt', 'jpg', 1))\n        if os.path.exists(gt_filepath):\n            with open(gt_filepath, 'r') as f:\n                for line in f:\n                    class_id, x_center, y_center, bbox_width, bbox_height = [\n                        float(x) for x in line.strip().split()\n                    ]\n\n                    # Convert to COCO format\n                    x_min = (x_center - (bbox_width / 2)) * 640  # image width\n                    y_min = (y_center - (bbox_height / 2)) * 640  # image height\n                    width = bbox_width * 640  # image width\n                    height = bbox_height * 640  # image height\n\n                    # Add annotation information\n                    coco_format['annotations'].append({\n                        'id': annotation_id,\n                        'image_id': image_id,\n                        'category_id': int(class_id),\n                        'bbox': [x_min, y_min, width, height],\n                        'area': width * height,\n                        'iscrowd': 0,\n                        'segmentation': [[0 for i in range(640)] for i in range(640)]\n                    })\n                    annotation_id += 1\n                    \n# Write the COCO format JSON to a file\nwith open(output_json_path, 'w') as f:\n    json.dump(coco_format, f, indent=2)\n","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.078166Z","iopub.status.idle":"2024-01-06T18:14:21.078510Z","shell.execute_reply.started":"2024-01-06T18:14:21.078339Z","shell.execute_reply":"2024-01-06T18:14:21.078355Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import json\n\n# Giả sử `ground_truths` là danh sách các dictionaries chứa thông tin ground truth\n# và `predictions` là danh sách các dictionaries chứa thông tin dự đoán.\n# Các dictionaries này phải có định dạng phù hợp với COCO.\n\n# Chuyển danh sách ground truths thành JSON\n# with open('ground_truths.json', 'w') as f:\n#     json.dump(tide_ground_truths, f)\n\n# Chuyển danh sách predictions thành JSON\nwith open('predictions.json', 'w') as f:\n    json.dump(tide_predictions, f)","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.079728Z","iopub.status.idle":"2024-01-06T18:14:21.080077Z","shell.execute_reply.started":"2024-01-06T18:14:21.079917Z","shell.execute_reply":"2024-01-06T18:14:21.079933Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"print(len(tide_predictions))\nprint(len(coco_format['images']))","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.081101Z","iopub.status.idle":"2024-01-06T18:14:21.081428Z","shell.execute_reply.started":"2024-01-06T18:14:21.081266Z","shell.execute_reply":"2024-01-06T18:14:21.081282Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"!pip install tidecv","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.083303Z","iopub.status.idle":"2024-01-06T18:14:21.083646Z","shell.execute_reply.started":"2024-01-06T18:14:21.083484Z","shell.execute_reply":"2024-01-06T18:14:21.083500Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"from tidecv import TIDE\nimport tidecv.datasets as datasets\n\npred = datasets.COCOResult('/kaggle/working/predictions.json')\ngt = datasets.COCO('/kaggle/working/ground_truths.json')","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.084563Z","iopub.status.idle":"2024-01-06T18:14:21.084906Z","shell.execute_reply.started":"2024-01-06T18:14:21.084742Z","shell.execute_reply":"2024-01-06T18:14:21.084758Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"from pycocotools.coco import COCO\nfrom pycocotools.cocoeval import COCOeval\n\n# Initialize COCO ground truth API\ncoco_gt = COCO('/kaggle/working/ground_truths.json')  # path to the JSON with ground truth annotations\n\n# Initialize COCO predictions API\ncoco_dt = coco_gt.loadRes('/kaggle/working/predictions.json')  # path to the JSON with detection results\n\n# Initialize COCOeval object\ncoco_eval = COCOeval(coco_gt, coco_dt, 'bbox')\n\n# Evaluate on a subset of images by setting their ids\n# If you want to evaluate on the entire validation set, you can omit this line\n# coco_eval.params.imgIds = image_ids  # list of image ids to evaluate on\n\n# Run evaluation\ncoco_eval.evaluate()\ncoco_eval.accumulate()\ncoco_eval.summarize()","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.085757Z","iopub.status.idle":"2024-01-06T18:14:21.086083Z","shell.execute_reply.started":"2024-01-06T18:14:21.085923Z","shell.execute_reply":"2024-01-06T18:14:21.085939Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"\ntide = TIDE()\n# Run the evaluations on the standard COCO metrics \ntide.evaluate_range(gt, pred, mode=TIDE.BOX)\n\ntide.summarize()","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.087380Z","iopub.status.idle":"2024-01-06T18:14:21.087746Z","shell.execute_reply.started":"2024-01-06T18:14:21.087555Z","shell.execute_reply":"2024-01-06T18:14:21.087571Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"tide.plot()","metadata":{"execution":{"iopub.status.busy":"2024-01-06T18:14:21.090311Z","iopub.status.idle":"2024-01-06T18:14:21.090668Z","shell.execute_reply.started":"2024-01-06T18:14:21.090487Z","shell.execute_reply":"2024-01-06T18:14:21.090514Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"    ","metadata":{},"execution_count":null,"outputs":[]}]}