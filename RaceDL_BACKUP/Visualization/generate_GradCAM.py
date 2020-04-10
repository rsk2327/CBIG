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

from scipy import imsave


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


from gradcam import *


##################

dataFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data'
truthFile = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/TargetFiles/TargetFile_Combined.csv'
modelFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/PyTorch/Models'

outputFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Visualization/GradCAM_Output/Model1'

savedModel = 'resnet50_1_11_74.0.pt'

imgSize = 256

##################





model = torch.load(os.path.join(modelFolder, savedModel))
model.eval()
model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(imgSize, imgSize))
gradcam = GradCAM(model_dict)


validTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
                   ToTensor(),
                   Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

validDf = outDf[outDf.train ==True]

validDataset = RaceDataset(dataFolder, validDf,validTransforms)
validLoader = torch.utils.data.DataLoader(validDataset, batch_size=valid_batch_size, 
										  shuffle=False)

iter = 0
for x,y in validLoader:
	x = x.numpy()

	mask, logit = gradcam(normed_img, class_idx=10)
	heatmap, cam_result = visualize_cam(mask, img)

	imsave(os.path.join(outputFolder, validDf.iloc[iter].filename))





