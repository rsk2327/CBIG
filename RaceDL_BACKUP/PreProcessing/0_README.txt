
################################

fileLocation_cluster.csv : 
--------------------------
Contains file locations for all the processed images in the cluster. This includes images from the initial dataset as well as the 3500 Caucasian women. 
* File contains full path name as well as just the image filename in another column.
* The images listed here are in jpeg format, the result of the preprocessing code applied to the original DICOM images.


Jpeg_conversion_script.ipynb :
----------------
Contains code to convert DICOM files to JPEG.
* Performs minimal processing to DICOM files
* Only the 4 key types : R_MLO_1, L_MLO_1, R_CC_1, L_CC_1 are retained. Rest images are not dropped
* Also contains code for helping with SFTP transfer of images from cluster to CBIG-DL system



createSymLink.py :
----------------
Used to create a clean, collated version of the entire dataset at RaceDL_BACKUP/Data.
All files listed in fileLocation_cluster.csv have been used to create symbolic links to the new output folder


