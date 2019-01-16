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



trainPath = '/home/santhosr/Documents/Birad/ProcessedData/FullRes_Sampled/'

data = (ImageItemList.from_folder(trainPath).random_split_by_pct(seed=40).label_from_folder().transform(get_transforms(),size=512).databunch(bs=15))

learn = create_cnn(data, models.resnet50, metrics=accuracy,pretrained=False)

best_model_cb = partial(ModelTrackerCallback)
learn.callback_fns.append(best_model_cb)



# learn.fit(10, 1e-2)
# learn.recorder.plot_losses()
# plt.savefig('train1.png')

# learn.fit(10, slice(1e-3,1e-2))
# learn.recorder.plot_losses()
# plt.savefig('train2.png')

# learn.fit(7, 1e-3)
# learn.recorder.plot_losses()
# plt.savefig('train3.png')

# learn.fit(7, slice(1e-4,1e-3))
# learn.recorder.plot_losses()
# plt.savefig('train4.png')


# learn.load('model_acc747_loss640')

# learn.freeze()
# learn.fit_one_cycle(15,1e-3)
# learn.recorder.plot_losses()
# plt.savefig('train5.png')

# learn.unfreeze()
# learn.fit(13, slice(1e-4,1e-3))
# learn.recorder.plot_losses()
# plt.savefig('train6.png')

learn.load('model_acc780_loss572')


learn.freeze()
#learn.fit_one_cycle(15,1e-2)
# learn.recorder.plot_losses()
# plt.savefig('train7.png')

learn.fit_one_cycle(15,1e-4)
learn.fit_one_cycle(15,1e-4)
learn.fit_one_cycle(15,1e-4)
learn.fit_one_cycle(15,1e-5)
learn.fit_one_cycle(15,1e-5)
# learn.recorder.plot_losses()
# plt.savefig('train8.png')