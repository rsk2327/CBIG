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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.misc import imsave

import pandas as pd
truthFile = '/home/santhosr/Documents/Birad/birad_targetFile.csv'
truth = pd.read_csv(truthFile)

trainPath = '/home/santhosr/Documents/Birad/ProcessedData/FullRes_RaceSampled/'

modelList = ['model_resnet_RS_id0_acc803_loss452',
'model_resnet_RS_id1_acc805_loss471',
'model_resnet_RS_id3_acc805_loss468',
'model_resnet_RS_id4_acc806_loss499',
'model_resnet_RS_id5_acc817_loss493',
'model_resnet_RS_id7_acc790_loss511',
'model_resnet_RS_id8_acc826_loss434',
'model_resnet_RS_id9_acc793_loss466',
'model_resnet_RS_id10_acc801_loss458',
'model_resnet_RS_id11_acc813_loss473',
'model_resnet_RS_id12_acc812_loss452',
'model_resnet_RS_id13_acc819_loss446']


seedList = [40,113,6767,999,555,1234,4321,1010,1101,1301,1701,2701]


def getTestAccuracy(model, data):
    
    predList2 = []
    target = []
    patientIDList = []
    viewList = []

    for i in tqdm(range(len(data.valid_ds.x))):
        p = model.predict(data.valid_ds.x[i])
        name = data.valid_ds.x.items[i].name
        
        if len(name.split("_"))==5:
            patientID = name.split("_")[1]
        else:
            patientID = name.split("_")[0]
        
        if 'MLO' in name:
            viewList.append('MLO')
        else:
            viewList.append('CC')
        
        predList2.append(p)
        target.append(int(data.valid_ds.y[i]))
        patientIDList.append(patientID)
    
    predClass = [int(i[1].numpy()) for i in predList2]
    
    df = pd.DataFrame({'target':target,'pred2':predClass,'patientID':patientIDList,'view':viewList})
    
    errors = len(df[df.target != df.pred2])
    acc = (len(target)-errors)/len(target)
    print("Validation accuracy : {}".format(acc))
    
    return df, acc


def checkPatientOverlap(data1,data2):
    
    patientIDList1 = []
    patientIDList2 = []
    
    for i in range(len(data1.x.items)):
        name = data1.x.items[i].name
        
        if len(name.split("_"))==5:
            patientID = name.split("_")[1]
        else:
            patientID = name.split("_")[0]
        
        patientIDList1.append(patientID)
        
    for i in range(len(data2.x.items)):
        name = data2.x.items[i].name
        
        if len(name.split("_"))==5:
            patientID = name.split("_")[1]
        else:
            patientID = name.split("_")[0]
        
        patientIDList2.append(patientID)
        
    patientIDList1 = set(patientIDList1)
    patientIDList2 = set(patientIDList2)
    
    commonPatients = list(patientIDList1.intersection(patientIDList2))
    
    patientIDList1 = list(patientIDList1)
    patientIDList2 = list(patientIDList2)
        
    
    return commonPatients, patientIDList1, patientIDList2
        

def getValidAccuracy( data, model=None, pred = None):
    
    if pred == None:
        pred = model.get_preds()
        
    common,trainList,validList = checkPatientOverlap(data.train_ds, data.valid_ds)
    inTrain=[]
    patientIDList=[]
    
    for i in range(len(data.valid_ds)):
        name = data.valid_ds.x.items[i].name
        
        if len(name.split("_"))==5:
            patientID = name.split("_")[1]
        else:
            patientID = name.split("_")[0]
        
        patientIDList.append(patientID)
        
        if patientID in trainList:
            inTrain.append(True)
        else:
            inTrain.append(False)
    
    target = pred[1].numpy()
    finalPred = np.argmax(pred[0].numpy(),1)
    score = accuracy_score(finalPred,target)
    
    df = pd.DataFrame({'target':target,'pred':finalPred,'ID':patientIDList,'inTrain':inTrain})
    
    totalNew = len(df[df.inTrain==False])
    totalNewCorrect = len(df[(df.inTrain==False) & (df.target == df.pred) ])
    newAccuracy = totalNewCorrect/totalNew
    
    
    print("Test images : {} Accuracy : {} Unseen Data Accuracy : {}".format(len(finalPred), score, newAccuracy))
    return score,df



for i in range(len(modelList)):

	trainData = (ImageItemList.from_folder(trainPath).random_split_by_pct(seed=seedList[i]).label_from_folder().transform(get_transforms(),size=512).databunch(bs=20).normalize())

	learn = create_cnn(trainData, models.resnet50, metrics=accuracy,pretrained=False)

	learn.load('/home/santhosr/Documents/Birad/ProcessedData/FullRes_RaceSampled/models/bestModels/'+modelList[i])

    print(modelList[i])
    
	# a,b,c = checkPatientOverlap(trainData.train_ds, trainData.valid_ds)
	
	# print("Common : {} Train : {} Valid : {}".format(len(a),len(b),len(c)))

	# testPath = '/home/santhosr/Documents/Birad/ProcessedData/FullRes_Test/'
	# testData = (ImageItemList.from_folder(testPath).random_split_by_pct(0.90,seed=40).label_from_folder().transform(get_transforms(),size=512).databunch(bs=20).normalize(trainData.stats))

	df,acc = getValidAccuracy(data = trainData,model = learn)
	# df.to_csv(modelList[i]+'.csv', index = False)

	# print("Accuracy : {}".format(acc))




