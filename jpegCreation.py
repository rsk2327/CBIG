# Code that reads in DCM files and converts them to JPEG images
# Also reads in the a truth file with binary labels for the DCM images.
# Based on their labels, the newly generated JPEG images are put into different folders



import os
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.misc import imsave
from imageio import imwrite
from skimage import img_as_ubyte

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


baseFolder = "/gpfs/cbica/projects/Prospr/Data/BreastAnatomy/dicoms"


folderList = os.listdir(baseFolder)

f = []
for folderName in folderList:
    if folderName.isdigit():
        f.append(folderName)
folderList = f


avoidList = [3800640, 4531207, 5248457, 75526133]
avoidList = [str(x) for x in avoidList]

for x in avoidList:
    folderList.remove(x)


truth = pd.read_csv('/gpfs/cbica/home/santhosr/Codes/Propsr_ClassLabels.csv')
truth = truth.loc[~truth.DummyID.isin(avoidList)]   #Removing the 4 bad cases

truth = truth.sort_values('DummyID')
truth.head()

logging.info('Truth file read. Number of records : {}'.format(len(truth)))


###### Preprocessing function

def preprocess(img):
    
    img = img + 15000
    
    shape = img.shape
    img = img.reshape((shape[0]*shape[1],))
    img = img.tolist()
    img = np.array([min(x,16384) for x in img])
    img = img.reshape(shape)
    
    return(img)

####### Converting and writing the JPEG files
count = 0
for i in tqdm(range(502,572)):
    
    if i>=len(truth):
        break
        
    folder = str(truth.iloc[i]['DummyID'])
    label = truth.loc[truth.DummyID==int(folder)]['Class'].values[0]

    
    img1 = pydicom.read_file(baseFolder + '/' + folder + "/" + folder + "_L_MLO_1.dcm").pixel_array
    img2 = pydicom.read_file(baseFolder + '/' + folder + "/" + folder + "_R_MLO_1.dcm").pixel_array

    img1 = preprocess(img1)
    img2 = preprocess(img2)
    
    
    
    if label == 0:
        
        outFolder = '/gpfs/cbica/home/santhosr/Datasets/jpeg/0'
        
        imwrite(outFolder  + "/" + folder + '_L_MLO_1.jpg', img1)
        imwrite(outFolder  + "/" + folder + '_R_MLO_1.jpg', img2)
        
    else:
        outFolder = '/gpfs/cbica/home/santhosr/Datasets/jpeg/1'
        
        imwrite(outFolder  + "/" + folder + '_L_MLO_1.jpg', img1)
        imwrite(outFolder  + "/" + folder + '_R_MLO_1.jpg', img2)
    count += 1
    
    logging.info('Moved {} . Total converted : {}'.format(folder, count))
    