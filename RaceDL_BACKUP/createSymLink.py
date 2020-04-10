import pandas as pd
import os
import sys

df = pd.read_csv('fileLocation_cluster.csv')

outputFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data/'

currentFileList = os.listdir(outputFolder)

for i in range(10):

	path = df.iloc[i]['fileLocation'][0]
	filename = df.iloc['filename']

	if filename in currentFileList:
		print("Already present")
		continue


	os.system(f'ln -s {path} {outputFolder}')