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


def createDataset(seed = 123,trainSplitRatio = 0.8, baseFolder = '/home/santhosr/Data/'):
    
    avoidList = [3800640, 4531207, 5248457, 75526133]
    avoidList = [str(x) for x in avoidList]

    truth = pd.read_csv('/home/santhosr/Documents/Codes/Propsr_ClassLabels.csv')
    truth = truth.loc[~truth.DummyID.isin(avoidList)]   #Removing the 4 bad cases
    
    np.random.seed(seed)

    #cancer patient list

    CPatientList = []

    fileList = os.listdir((os.path.join(baseFolder,"C")))

    for i in range(len(fileList)):
        CPatientList.append(fileList[i].split("_")[0])

    CPatientList = list(set(CPatientList))

    #Non cancer patient list

    NCPatientList = []

    fileList = os.listdir((os.path.join(baseFolder,"NC")))

    for i in range(len(fileList)):
        NCPatientList.append(fileList[i].split("_")[0])

    NCPatientList = list(set(NCPatientList))

    fullPatientList = CPatientList + NCPatientList
    
    #Select separate patient lists for train and validation
    trainSize = int(trainSplitRatio*len(fullPatientList))

    trainPatientList = list(np.random.choice(fullPatientList,trainSize,replace=False))

    testPatientList = list(set(fullPatientList).difference(set(trainPatientList)))
    
    #Creating file lists

    trainFileList = []

    for i in range(len(trainPatientList)):
        trainFileList.append(trainPatientList[i]+'_L_MLO_1.jpg')
        trainFileList.append(trainPatientList[i]+'_R_MLO_1.jpg')

    testFileList = []

    for i in range(len(testPatientList)):
        testFileList.append(testPatientList[i]+'_L_MLO_1.jpg')
        testFileList.append(testPatientList[i]+'_R_MLO_1.jpg')
        
    trainInd = [False]*len(trainFileList) + [True]*len(testFileList)
    
    df = pd.DataFrame({'files':trainFileList + testFileList, 'train':trainInd})
    
    df['DummyID'] = df['files'].apply(lambda x : int(x.split("_")[0]))
    df = pd.merge(df,truth,on='DummyID',how='left')
    
    #Upsampling class for train dataset
    upSampleCount = 4

    df2 = df.copy()

    for i in range(upSampleCount):
        df2 = pd.concat([df2, df.loc[(df.train==False) & (df.Class==1)]])
    df = df2.copy()
    
    df.drop(['DummyID','Class','MatchedGroup'],axis=1,inplace = True)
    
    return df
    
                      

def getLabel(x,truth,binary=False):
    ID = x.split("/")[-1].split("_")[0]
    label = truth[truth.DummyID == int(ID)]['Class'].values[0]

    return(label)