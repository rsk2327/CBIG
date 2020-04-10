import torch
import torchvision

import os
import pandas as pd 
import numpy as np 
import sys
import logging
import time
import copy
import logging
import warnings
warnings.filterwarnings('ignore')



from tqdm import tqdm
from torch.nn import Sigmoid
from torch import Tensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection.image_list import ImageList 
from torchvision.transforms import Resize,ToTensor, RandomHorizontalFlip, RandomVerticalFlip,Normalize
from torchvision import models
from torch.optim import lr_scheduler

from progressbar import progressbar

from PIL import Image



##################

dataFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data'
truthFile = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/TargetFiles/TargetFile_Combined.csv'
modelFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/PyTorch/Models'

savedModel = 'resnet50_1_11_74.0.pt'


##################



# initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(256, 256))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)