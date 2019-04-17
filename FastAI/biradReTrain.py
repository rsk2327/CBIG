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

data = (ImageItemList.from_folder(trainPath).random_split_by_pct(seed=321).label_from_folder().transform(get_transforms(),size=1024).databunch(bs=5))


##### CALLBACK
class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', prefix:str='resnet50',id:int=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        print(epoch)
        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            self.learn.save(f'model_'+str(int)+f'_resnet50_acc{int(acc*1000)}_loss{int(val_loss*1000)}')


##### TRAINING

learn = create_cnn(data, models.resnet50, metrics=accuracy,pretrained=False)
best_model_cb = partial(ModelTrackerCallback,id=9)
learn.callback_fns.append(best_model_cb)

# learn.model[-1] = create_head(nf=4096,nc=4)
# learn.model[-1].cuda()

# learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/'+'model_resnet_acc893_loss254')

# learn.load('/home/santhosr/Data/models/'+'model_resnet50_acc698_loss627')


# learn.model[-1] = create_head(nf=4096,nc=2)
# learn.model[-1].cuda()

learn.data = data

# STEP 1
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
learn.fit_one_cycle(10,1e-5)
learn.fit(20,1e-5)


# learn.freeze()
# learn.fit(10,1e-4)
# learn.unfreeze()
# learn.fit(10,1e-5)
# learn.fit(10,1e-5)
# learn.freeze(),
# learn.fit(10,1e-4)
# learn.unfreeze()
# learn.fit_one_cycle(10,1e-5)
# learn.fit(20,1e-5)