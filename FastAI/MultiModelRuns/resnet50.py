import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
import torchvision.models as tmodels

import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.misc import imsave

class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, id:int , monitor:str='val_loss', mode:str='auto', prefix:str='resnet50'):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.id = id
        self.bestAcc = 0.0001
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."

        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            if acc>0.79:
                self.learn.save(f'model_resnet_RS_id{self.id}_acc{int(acc*1000)}_loss{int(val_loss*1000)}')



trainPath = '/home/santhosr/Documents/Birad/ProcessedData/FullRes_RaceSampled/'



def runTest(id = 1, seed = 40):


    data = (ImageItemList.from_folder(trainPath).random_split_by_pct(seed=seed).label_from_folder().transform(get_transforms(),size=512).databunch(bs=20).normalize())

    learn = create_cnn(data, models.resnet50, metrics=accuracy,pretrained=False)

    best_model_cb = partial(ModelTrackerCallback,id=id)
    learn.callback_fns.append(best_model_cb)

    learn.unfreeze()
    print(time.ctime())
    print("TRAINING START")
    learn.fit_one_cycle(15,1e-3)
    learn.fit_one_cycle(15,1e-3)
    learn.fit_one_cycle(15,1e-3)
    learn.fit_one_cycle(15,1e-4)
    learn.fit_one_cycle(15,1e-4)
    print(time.ctime())
    print("TRAINING END")


# runTest(id=1,seed=113)
# runTest(id=2,seed=1683)
# runTest(id=3,seed=6767)
# runTest(id=4,seed=999)
# runTest(id=5,seed=555)

# runTest(id=6,seed=888)

# runTest(id=7,seed=1234)
# runTest(id=8,seed=4321)

# runTest(id=9,seed=1010)
# runTest(id=10,seed=1101)
# runTest(id=11,seed=1301)
# runTest(id=12,seed=1701)

runTest(id=13,seed=2701)
# runTest(id=13,seed=2701)
