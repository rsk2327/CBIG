import os
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.misc import imsave, imresize
from imageio import imwrite,imread

from tqdm import tqdm

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)




baseFolder = '/gpfs/cbica/projects/Mayo-Radiomics/Data/Penn/sorted_2/'
outputFolder = '/gpfs/cbica/home/santhosr/CBIG/BIRAD/ProcessedData/inverted'
truthFile = '/gpfs/cbica/home/santhosr/CBIG/BIRAD/birad_targetFile.csv'


def preprocess(img):
    
    out_img = img.astype(np.float32).copy()

    if np.min(out_img) < 1:
        out_img = out_img + np.absolute(np.min(out_img)) + 1
    out_img = np.log(np.log(np.log(np.log(out_img))))
    
    out_img = np.absolute(out_img - np.max(out_img))

    return(out_img)

truth = pd.read_csv(truthFile)

folderList = os.listdir(baseFolder)
errorFiles = []

for j in tqdm(range(500,1500)):

	fileList = os.listdir(os.path.join(baseFolder, folderList[j]))


	for i in range(len(fileList)):
	    
	    fileName= fileList[i]
	    
	    if '.dcm' in fileName:
	        
	        if 'R_MLO_1' in fileName or 'L_MLO_1' in fileName or 'R_CC_1' in fileName or 'L_CC_1' in fileName:
	            
	            img = pydicom.read_file(os.path.join(baseFolder, folderList[j], fileName)).pixel_array
	            img = preprocess(img)
	            
	            
	            outFileName = fileName.split(".")[0]+'.jpg'
	            patientName = fileName.split("_")[0]
	            
	            
	            label = truth.loc[truth.DummyID == int(patientName)]['Density_Overall'].values[0]
	            
	            
	            try:
	                imwrite(os.path.join(outputFolder, str(label),outFileName), img)
	            except:
	                print(outFileName)
	                errorFiles.append(outFileName)