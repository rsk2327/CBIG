import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *

import pandas as pd
import numpy as np
import os
from fastai.callbacks import *

trainPath = '/home/santhosr/Data/'

data = (ImageItemList.from_folder(trainPath).random_split_by_pct(seed=40).label_from_folder().transform(get_transforms(),size=1024).databunch(bs=5))


##### CALLBACK
class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', prefix:str='resnet50'):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            self.learn.save(f'model_acc{int(acc*1000)}_loss{int(val_loss*1000)}')


##### TRAINING

learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.load('/home/santhosr/Data/models/model_acc790_loss502')

best_model_cb = partial(ModelTrackerCallback)
learn.callback_fns.append(best_model_cb)





learn.fit_one_cycle(15,0.0001)

learn.fit_one_cycle(25,0.0001)

learn.fit_one_cycle(25,0.00001)