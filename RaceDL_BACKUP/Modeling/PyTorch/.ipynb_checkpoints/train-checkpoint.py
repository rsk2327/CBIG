import torch
import torchvision

import os
import pandas as pd 
import numpy as np 
import sys
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


from PIL import Image


dataFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data'



class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataFolder, dataList):
    
    
    self.data = dataList


    
  def __len__(self):
    
    return len(self.data)
    
  def __getitem__(self, index):
    
    filename = self.data.iloc[index].filename
    
    if 'FullRes' in filename :
        #Removes the starting "FullRes" from the filename
        filename = filename[7:]
    
    data = imread(os.path.join(dataFolder, filename))
    
    
    data = Image.fromarray(data)
    data = data.resize((512,512))
    
    data = np.array(data)
    
    
    
    
    
    data = data/256.0 
    
    data = torchvision.transforms.ToTensor()(data)
    
    data = data.float()
    
    label = self.data.iloc[index].label
    bmi = self.data.iloc[index].BMI
    
   
    return data,label, bmi

