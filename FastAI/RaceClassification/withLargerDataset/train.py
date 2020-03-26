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

sys.path.insert(0, '/home/santhosr/Documents/Birad/FastAI/RaceClassification/withLargerDataset')

from utils import *

from birad import *

########### INPUT PARAMS #########################
dataFileName = 'DataFile10.csv'
modelID = 10
numPatients = 5000




##################################################






inputFolder1 = '/home/santhosr/Documents/Birad/ProcessedData/FullRes'
truthFile1 = '/home/santhosr/Documents/Birad/birad_targetFile.csv'

inputFolder2 = '/home/santhosr/Documents/Birad/ProcessedData/PennExtra_3500/'
truthFile2 = '/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv'

df1 = pd.read_csv('/home/santhosr/Documents/Birad/birad_targetFile.csv')
df1.drop(['PresIntentType','DBT'],inplace = True,axis=1)


df2 = pd.read_csv('/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv')
df2.Medview_Race = 'White'
truth = pd.concat([df1,df2],sort=True)









# df = createDataset2(seed=123,n=numPatients  )
df = createBMIDataset()
# df = upSampleDf(df)
df= df.sample(frac=1)

df.to_csv(dataFileName ,index=False,index_label=False)

# df = pd.read_csv("DataFile.csv")




def getRaceLabel(x,binary=False):
    ID = x.split("/")[-1].split("_")[0]
    label = truth[truth.DummyID == int(ID)]['Medview_Race'].values[0]

    if label == 'African American':
        return 0
    elif label == "White":
        return 1
    else:
        return 2
    



#Creates the FastAI Dataset
data = ImageItemList.from_df(df=df,path='/home/santhosr/Documents/Birad/ProcessedData/', cols='filename').split_from_df(col='train').label_from_func(getRaceLabel).transform(get_transforms(),size=256).databunch(bs=50).normalize()
print("Dataset created")


#Creates the model architecture 
learn = create_cnn(data, tmodels.resnet50, metrics=accuracy,pretrained=True)

# learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/model_resnet50_acc668_loss600')

best_model_cb = partial(ModelTrackerCallback,id=modelID, modelName = "resnet50")
learn.callback_fns.append(best_model_cb)

#Unfreezes the entire model
learn.unfreeze()

#Uses a fixed learning rate of 1e-5
learn.fit(30,1e-5)

learn.unfreeze()
learn.fit_one_cycle(10,slice(1e-5))
learn.fit_one_cycle(10,slice(1e-5))
learn.fit(5,slice(1e-4))

learn.freeze()
learn.fit(3,1e-4)

learn.unfreeze()
learn.fit_one_cycle(5,1e-6)
learn.fit(5,1e-5)