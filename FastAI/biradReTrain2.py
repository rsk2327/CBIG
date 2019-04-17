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

class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', prefix:str='resnet50',id:int=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        self.id = id
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        print(epoch)
        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            self.learn.save(f'model_'+str(self.id)+f'_resnet50_acc{int(acc*1000)}_loss{int(val_loss*1000)}')
            

sys.path.insert(0, '/home/santhosr/Documents/Codes/FastAI/utils.py' )

from utils import *



global truth
avoidList = [3800640, 4531207, 5248457, 75526133]
avoidList = [str(x) for x in avoidList]

truth = pd.read_csv('/home/santhosr/Documents/Codes/Propsr_ClassLabels.csv')
truth = truth.loc[~truth.DummyID.isin(avoidList)]   #Removing the 4 bad cases

df = createDataset(seed = 111)

getLabel = partial(getLabel,truth = truth)

data = ImageItemList.from_df(df=df,path='/home/santhosr/Data_Combined/', cols='files').split_from_df(col='train').label_from_func(getLabel).transform(get_transforms(),size=1024).databunch(bs=5).normalize()
print("Dataset created")

learn = create_cnn(data, models.resnet50, metrics=accuracy,pretrained=True)


learn.load('/home/santhosr/Data_Combined/models/model_1_resnet50_acc630_loss754')
best_model_cb = partial(ModelTrackerCallback,id=1)
learn.callback_fns.append(best_model_cb)

learn.freeze()
learn.fit(10,1e-4)
learn.unfreeze()
learn.fit(10,1e-5)
learn.fit(10,1e-5)
learn.fit_one_cycle(10,1e-5)
learn.fit(20,1e-5)


learn.freeze()
learn.fit(10,1e-4)
learn.unfreeze()
learn.fit(10,1e-5)
learn.fit(10,1e-5)
learn.freeze(),
learn.fit(10,1e-4)
learn.unfreeze()
learn.fit_one_cycle(10,1e-5)
learn.fit(20,1e-5)