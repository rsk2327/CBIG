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

from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.misc import imsave


import os
import sys

sys.path.insert(0, '/home/santhosr/Documents/Birad/FastAI/RaceClassification/' )

from utils import *

from birad import *
from birad import setTruthFile, getRaceLabel



inputFolder1 = '/home/santhosr/Documents/Birad/ProcessedData/FullRes'
truthFile1 = '/home/santhosr/Documents/Birad/birad_targetFile.csv'

inputFolder2 = '/home/santhosr/Documents/Birad/ProcessedData/PennExtra_3500/'
truthFile2 = '/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv'

df1 = pd.read_csv('/home/santhosr/Documents/Birad/birad_targetFile.csv')
df1.drop(['PresIntentType','DBT'],inplace = True,axis=1)


df2 = pd.read_csv('/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv')
df2.Medview_Race = 'White'
truth = pd.concat([df1,df2],sort=True)

setTruthFile(truth)


dfFile = "DataFile10.csv"

modelName = 'model_resnet50_id10_acc833_loss386'


#Feature Directories
trainFolder = '/home/santhosr/Documents/Birad/FastAI/RaceClassification/withLargerDataset/Features/model9_2/train'

validFolder = '/home/santhosr/Documents/Birad/FastAI/RaceClassification/withLargerDataset/Features/model9_2/valid'


df = pd.read_csv(dfFile)




#Creates the FastAI Dataset
data = ImageItemList.from_df(df=df,path='/home/santhosr/Documents/Birad/ProcessedData/', cols='filename').split_from_df(col='train').label_from_func(getRaceLabel).transform(get_transforms(),size=256).databunch(bs=50).normalize()
print("Dataset created")



#Creates the model architecture 
learn = create_cnn(data, tmodels.resnet50, metrics=accuracy,pretrained=True)

learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/'+modelName)



predList = []

for i in tqdm(range(len(data.train_ds))):
    pred = learn.predict(data.train_dl.x[i])
    predList.append(pred)





predScores = []

for i in range(len(predList)):
    predScores.append([predList[i][2].numpy()[0],predList[i][2].numpy()[1]])

predLabels = []

for i in range(len(predList)):
    predLabels.append(int(predList[i][1].numpy()))

imageNames = []

for i in range(len(data.train_ds)):
    imageNames.append( data.train_ds.items[i].split("/")[-1].split(".")[0]  )


predDf = pd.DataFrame(predScores)
predDf.columns = ['score0','score1']

predDf['predLabel'] = predLabels

predDf['imageName'] = imageNames

predDf['truthLabel'] = predDf.imageName.apply(lambda x : getRaceLabel(x))

print("Accuracy Score :")
print(accuracy_score(predDf.predLabel, predDf.truthLabel))


predDf.to_csv('model10_1_TrainPredictions.csv',index = False, index_label = False)

