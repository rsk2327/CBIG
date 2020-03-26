
# coding: utf-8

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
import torchvision.models as tmodels

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.misc import imsave

import os
import sys

sys.path.insert(0, '/home/santhosr/Documents/Birad/FastAI/RaceClassification/' )

from utils import *

from birad.blackbox import *
from birad import *


######### INPUT PARAMS ########################################

dfFile = "DataFile9.csv"

modelName = 'model_resnet50_id9_acc848_loss376'


#Feature Directories
trainFolder = '/home/santhosr/Documents/Birad/FastAI/RaceClassification/withLargerDataset/Features/model9_1/train'

validFolder = '/home/santhosr/Documents/Birad/FastAI/RaceClassification/withLargerDataset/Features/model9_1/valid'



###############################################################



inputFolder1 = '/home/santhosr/Documents/Birad/ProcessedData/FullRes'
truthFile1 = '/home/santhosr/Documents/Birad/birad_targetFile.csv'

inputFolder2 = '/home/santhosr/Documents/Birad/ProcessedData/PennExtra_3500/'
truthFile2 = '/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv'

df1 = pd.read_csv('/home/santhosr/Documents/Birad/birad_targetFile.csv')
df1.drop(['PresIntentType','DBT'],inplace = True,axis=1)


df2 = pd.read_csv('/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv')
df2.Medview_Race = 'White'
truth = pd.concat([df1,df2],sort=True)




df = pd.read_csv(dfFile)

setTruthFile(truth)

    



#Creates the FastAI Dataset
data = ImageItemList.from_df(df=df,path='/home/santhosr/Documents/Birad/ProcessedData/', cols='filename').split_from_df(col='train').label_from_func(getRaceLabel).transform(get_transforms(),size=256).databunch(bs=50).normalize()
print("Dataset created")

# ##### Loading the model


#Creates the model architecture 
learn = create_cnn(data, tmodels.resnet50, metrics=accuracy,pretrained=True)

learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/'+modelName)


generateFeatures(learn,data, trainFolder, validFolder)



# learn.model.eval()

# #Setting the layer from which we are extracting features
# layer = list(learn.model.children())[1][4]

# my_embedding = 0

# def copyData(m, inp, out):
#     global my_embedding
#     out1 = out.detach().cpu().numpy()
#     my_embedding = out1

# #Registering a forward hook
# feat = layer.register_forward_hook(copyData)


# # ##### Extracting Features (Train Data)

# ### Train Data

# for i in tqdm(range(len(data.train_ds.items))):
    
#     e=data.one_item(data.train_ds.x[i])
#     pred = learn.model(e[0])
    
#     file = data.train_ds.items[i].split("/")[-1].split(".")[0]
    
#     np.save( os.path.join(trainFolder, file+'.npy'), my_embedding )
    
    
    
    


# #### Extracting Features (Valid Data)

# ### Valid Data

# for i in tqdm(range(len(data.valid_ds.items))):
    
#     e=data.one_item(data.valid_ds.x[i])
#     pred = learn.model(e[0])
    
#     file = data.valid_ds.items[i].split("/")[-1].split(".")[0]
    
#     np.save( os.path.join(validFolder, file+'.npy'), my_embedding )
    
    
    
    

