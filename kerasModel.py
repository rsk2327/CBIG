import pandas as pd
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt

from scipy.misc import imsave
from imageio import imwrite, imread
from tqdm import tqdm
from skimage import *
import itertools


%matplotlib inline


import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from math import sqrt
from keras.callbacks import History 
from keras.optimizers import Adam, SGD
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.layers import BatchNormalization
from keras.models import Model
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
from math import sqrt
from keras import backend as K
from keras.callbacks import History 
import gc


from utils import *

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batchSize = 25
imageSize = (512,512)

train_flow = train_datagen.flow_from_directory('/gpfs/cbica/home/santhosr/Datasets/jpeg',batch_size=batchSize,target_size=imageSize)


def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1/10)(bn)
    return act


### MODEL DEFINITION

inp = Input(shape=(imageSize[0], imageSize[1],3))

conv1 = conv_layer(inp, 64, zp_flag=False)
conv2 = conv_layer(conv1, 64, zp_flag=False)
mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
# 23
conv3 = conv_layer(mp1, 128, zp_flag=False)
conv4 = conv_layer(conv3, 128, zp_flag=False)
conv4_2 = conv_layer(conv4, 128, zp_flag=False)
mp2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
9
conv7 = conv_layer(mp2, 256, zp_flag=False)
conv8 = conv_layer(conv7, 256, zp_flag=False)
conv9 = conv_layer(conv8, 256, zp_flag=False)
mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)

conv10 = conv_layer(mp3, 256, zp_flag=False)
conv11 = conv_layer(conv10, 256, zp_flag=False)
mp4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv11)

conv12 = conv_layer(mp4, 256, zp_flag=False)
conv13 = conv_layer(conv12, 256, zp_flag=False)
mp5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv13)



# dense layers
flt = Flatten()(mp5)

ds1 = Dense(64, activation='relu')(flt)
ds2 = Dense(64, activation='relu')(ds1)
out = Dense(2, activation='softmax')(ds1)

model = Model(inp, out)


model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy',f1])

epochs = 10

model.fit_generator(train_flow,epochs=epochs,verbose=1)