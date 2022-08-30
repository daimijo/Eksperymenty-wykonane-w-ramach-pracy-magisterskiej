#Program ustworzony na podstawie szablonu ze strony https://github.com/jacobgil/pytorch-grad-cam

import io
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import pandas
import torchvision
model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
from PIL import Image
import requests
from io import BytesIO
import sys
import numpy as np
import torch
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
from PIL import Image
img = Image.open("images/goldfish.png")
img_rgb = np.asarray(img) / 255

input_tensor = transform(img)
input_tensor = input_tensor.unsqueeze(0)

model.eval()

out =  model(input_tensor)
with open('imagenet1000_clsidx_to_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
# Construct the CAM object once, and then re-use it on many images:
#cam = ScoreCAM(model=model, target_layers=target_layers)
cam = GradCAM(model=model, target_layers=target_layers)
#cam = ScoreCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

im = Image.fromarray(visualization)
im.show()