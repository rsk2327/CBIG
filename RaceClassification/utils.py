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

import pandas as pd
import numpy as np





def createDataset2(n=600,seed = 123,full=False):

    patientList = []
    fullFileList = []

    inputFolder1 = '/home/santhosr/Documents/Birad/ProcessedData/FullRes'
    truthFile1 = '/home/santhosr/Documents/Birad/birad_targetFile.csv'

    inputFolder2 = '/home/santhosr/Documents/Birad/ProcessedData/PennExtra_3500/'
    truthFile2 = '/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv'

    df1 = pd.read_csv('/home/santhosr/Documents/Birad/birad_targetFile.csv')
    df1.drop(['PresIntentType','DBT'],inplace = True,axis=1)


    df2 = pd.read_csv('/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv')
    df2.Medview_Race = 'White'
    
    ## Removing IDs from df2 which are already present in df1
    idList = list(df1.DummyID.values)
    df2 = df2[~df2.DummyID.isin(idList)]
    
    truth = pd.concat([df1,df2],sort=True)

    ## Reading from set 1
    for i in range(1,5):

        folder = os.path.join(inputFolder1,str(i))
        fileList = os.listdir(folder)
        fileList = [os.path.join('FullRes',str(i),x) for x in fileList]
        fullFileList = fullFileList + fileList
        print(len(fileList))

        patientList = patientList + [int(x.split("/")[-1].split("_")[0]) for x in fileList]
    
    patientList1 = patientList.copy()
    ## Reading from set 2
    print(len(patientList))
    
    fileList= os.listdir(inputFolder2)
    fileList = [os.path.join('PennExtra_3500',x) for x in fileList]
    d = pd.DataFrame(fileList)
    d[1] = d[0].apply(lambda x : int(x.split("/")[1].split("_")[0]))
    d = d[d[1].isin(df2.DummyID.values)]
    fileList = list(d[0].values)
    fullFileList += list(d[0].values)
    
    patientList += [int(x.split("/")[-1].split("_")[0]) for x in fileList]
    print(len(patientList))
    
    patientList2 = patientList.copy()
    
    #Retaining only the patients with 4 views
    k=pd.Series(patientList).value_counts().reset_index()
    patientList = k[k[0]==4]['index'].values
    print("total number of patients",len(patientList))

    patientList = np.array(list(set(patientList)))
    df = pd.DataFrame({'DummyID':patientList})
    df = pd.merge(df,truth, how='left')
    df1 = df1.copy()
    df = df.drop_duplicates(subset=['DummyID'])
    
    



    #Creates equal number of patients from White and AA groups
    white = df[df.Medview_Race=='White']
    AA = df[df.Medview_Race=='African American']

    white = white.sort_values('DummyID')
    AA = AA.sort_values('DummyID')

    #Randomly selects n patients from each group
    whiteOverallList = np.random.choice(white.DummyID, n,replace = False)
    AAOverallList = np.random.choice(AA.DummyID, n,replace = False)

    #Training datasets (List of patients for training)
    whiteTrainList = np.random.choice(whiteOverallList, int(0.8*n), replace = False)
    AATrainList = np.random.choice(AAOverallList, int(0.8*n), replace = False)

    #Validation datasets (List of patients for validation)
    whiteValidList = np.array(list(set(whiteOverallList).difference(set(whiteTrainList))))
    AAValidList = np.array(list(set(AAOverallList).difference(set(AATrainList))))

    trainPatientList = np.concatenate([whiteTrainList, AATrainList])
    validPatientList = np.concatenate([whiteValidList, AAValidList])

    temp = pd.DataFrame(fullFileList)
    temp.columns = ['filename']

    temp['DummyID'] = temp.filename.apply(lambda x : int(x.split("/")[-1].split("_")[0]))
                                          
    trainTemp = temp[temp.DummyID.isin(trainPatientList)]
    validTemp = temp[temp.DummyID.isin(validPatientList)]

    trainTemp['train'] = False
    validTemp['train'] = True

    df= pd.concat([trainTemp, validTemp], sort = True)

    #Shuffling data
    index = list(range(len(df)))
    np.random.shuffle(index)
    df = df.iloc[index]
                                          
    return df









class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, path:str='/home/santhosr/Documents/Birad/ProcessedData/models',id:int=None,monitor:str='val_loss', mode:str='auto',modelName:str='resnet50'):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        self.folderPath = path
        self.id = id
        self.modelName = modelName
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."

        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            if self.id==None:
                fileName = 'model_'+self.modelName+'_acc'+str(int(acc*1000))+"_loss"+str(int(val_loss*1000))
            else:
                fileName = 'model_'+self.modelName+'_id' + str(self.id) + '_acc' + str(int(acc*1000)) + "_loss" + str(int(val_loss*1000))
            fileName = os.path.join(self.folderPath, fileName)
            self.learn.save(fileName)    