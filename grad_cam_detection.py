#Program utworzony na podstawie tutoriala dostępnego na stronie https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import requests
import torchvision
from PIL import Image

def predict(input_tensor, model, blank, detection_threshold):
    outputs = model(input_tensor)
    prediction_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    prediction_labels = outputs[0]['labels'].cpu().numpy()
    prediction_scores = outputs[0]['scores'].detach().cpu().numpy()
    prediction_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    bboxes, classes, labels, indices = [], [], [], []
    for i in range(len(prediction_scores)):
        if prediction_scores[i] >= detection_threshold:
            bboxes.append(prediction_bboxes[i].astype(np.int32))
            classes.append(prediction_classes[i])
            labels.append(prediction_labels[i])
            indices.append(i)
    bboxes = np.int32(bboxes)
    return bboxes, classes, labels, indices


def draw_boxes(bboxes, labels, classes, image):
    for i, bbox in enumerate(bboxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(bbox[0]), int(bbox[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'blank',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'blank', 'backpack', 'umbrella',
              'blank', 'blank', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'blank', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'home_plant', 'bed', 'blank', 'table', 'blank', 'blank', 'toilet',
              'blank', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'baking_oven', 'toaster', 'sink', 'refrigerator', 'blank', 'book', 'clock', 'vase',
              'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
image_url = "https://www.swiatobrazu.pl/picture/i/2g9/x70/u32/cfccf_s_660_0_0_0_0_0_0_2g9x70u3278gu2qrimq15hj3b4b2kxec1eyw4isyfr40jkhryyit5sm6mozy93z1.jpeg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw))
image_float_np = np.float32(image) / 255
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

input_tensor = transform(image)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
input_tensor = input_tensor.unsqueeze(0)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)
bboxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(bboxes, labels, classes, image)
image_final = Image.fromarray(image)
image_final.show()
target_layers = [model.backbone]
targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=bboxes)]
cam = EigenCAM(model,
               target_layers,
               use_cuda=torch.cuda.is_available(),
               reshape_transform=fasterrcnn_reshape_transform)

grayscale_cam = cam(input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

image_with_bounding_boxes = draw_boxes(bboxes, labels, classes, cam_image)

image_final = Image.fromarray(image_with_bounding_boxes)
image_final.show()
def renormalize(bboxes, image_alt, grayscale_cam):
    norm_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in bboxes:
        img = norm_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    norm_cam = np.max(np.float32(images), axis=0)
    norm_cam = scale_cam_image(norm_cam)
    eigencam_image_renormalized = show_cam_on_image(image_alt, norm_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(bboxes, labels, classes, eigencam_image_renormalized)
    return image_with_bounding_boxes


image_final = Image.fromarray(renormalize(bboxes, image_float_np, grayscale_cam))
image_final.show()