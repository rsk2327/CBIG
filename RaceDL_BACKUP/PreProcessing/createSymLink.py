import pandas as pd
import os
import sys

df = pd.read_csv('fileLocation_cluster.csv')

outputFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data/'

currentFileList = os.listdir(outputFolder)

os.chdir(outputFolder)

for i in range(len(df)):

	if i%1000 == 0 :
		print(i)
		print("----")
		
	path = df.iloc[i]['fileLocation']
	filename = df.iloc[i]['filename']


	if filename+'.jpg' in currentFileList:
		continue


	os.system(f'ln -s {path} ')