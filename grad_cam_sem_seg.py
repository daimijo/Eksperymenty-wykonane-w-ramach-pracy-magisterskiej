#Program utworzony na podstawie tutoriala ze strony https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.ipynb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt

#image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
#image_url = "https://www.yourlegalfriend.com/siteassets/services/accidents--illness/road-accidents/car-accidents/cars-on-the-road.jpg"
#image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_hW2h96nMsDfdMRCoXpbCkIcGG1ZWKVFGBg&usqp=CAU.jpg"
#image_url = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/203313991/1800"
#image_url = "https://ichef.bbci.co.uk/news/660/cpsprodpb/D88A/production/_114843455_gettyimages-1216452636.jpg"
#image_url = "https://gulfcoastbirdobservatory.files.wordpress.com/2013/10/chimney-swift-nest.jpg"
#image_url = "https://img.buzzfeed.com/buzzfeed-static/static/2020-06/22/15/asset/95e6b38c16ff/sub-buzz-11915-1592840455-17.jpg?resize=625:938g"
#image_url = "https://www.birdspot.co.uk/wp-content/uploads/2020/11/magnificent-frigatebird.jpg"
#image_url="https://media.istockphoto.com/photos/bird-sparrow-on-white-background-picture-id668770244?k=20&m=668770244&s=612x612&w=0&h=DM4iiHD53Lkvg7GCSX8KmMy1hV9CyL__MWd_bBMrUG4="
#image_url = "https://static.photocrowd.com/upl/2W/cms.qixgkNQouRcMYxhX3mKQ-hd.jpeg"
image_url = "https://images.ctfassets.net/cnu0m8re1exe/1ep16TzpUzqn7rBBhS1YkR/47e40d3b04c64b36c0455e07aee7e319/swifts.jpg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw))
rgb_img = np.float32(image) / 255
input = preprocess_image(rgb_img,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
# Taken from the torchvision tutorial
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
model = deeplabv3_resnet50(pretrained=True, progress=False)
model = model.eval()
output = model(input)

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


model = SegmentationModelOutputWrapper(model)
output = model(input)

masks = torch.nn.functional.softmax(output, dim=1).cpu()
classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}

bird_category = class_to_idx["bird"]
bird_mask = masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
bird_mask_uint8 = 255 * np.uint8(bird_mask == bird_category)
bird_mask_float = np.float32(bird_mask == bird_category)

both_images = np.hstack((image, np.repeat(bird_mask_uint8[:, :, None], 3, axis=-1)))
image_2 = Image.fromarray(both_images)
image_2.show()
from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)


    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


target_layers = [model.model.backbone.layer4]
targets = [SemanticSegmentationTarget(bird_category, bird_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

image_2 = Image.fromarray((cam_image))
image_2.show()