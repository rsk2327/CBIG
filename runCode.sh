#!/bin/bash
############################ Stadard parameters to the "qsub" command ##########
#### Set the shell (under SGE).
#$ -S /bin/bash
#### Run the commands in the directory where the SGE "qsub" command was given:
#$ -cwd
#### Joint the stdout and stderr together.
#$ -j y
#### save the standard output. By default, the output will be saved into your
#### home directory. The "-o" option lets you specify an alternative directory.
#$ -o $HOME/sge_job_output/$JOB_NAME.$JOB_ID.log
#### save the standard error:
#$ -e $HOME/sge_job_output/$JOB_NAME.$JOB_ID.log
####
#### My email address:
#$ -M somilgo@seas.upenn.edu
###$ -m b #### send mail at the beginning of the job
###$ -m e #### send mail at the end of the job
#$ -m a #### send mail in case the job is aborted
##################################
#### Optional SGE "qsub" parameters that could be used to customize
#### the submitted job. In each case, remove the string:
####		REMOVE_THIS_STRING_TO_ENABLE_OPTION
#### but leave the characters:
#### 		#$
#### at the beginning of the line.#! /bin/bash
####
### Indicate that the job is short, and will complete in under 15 minutes so
### that SGE can give it priority.
### 	WARNING! If the job takes more than 15 minutes it will be killed.
####$  -l short
####
#### Request that the job be given 6 "slots" (CPUS) on a single server instead
#### of 1. You MUST use this if your program is multi-threaded, you should NOT
#### use it otherwise. Most jobs are not multi-threaded and will not need this
#### option.
#REMOVE_THIS_STRING_TO_ENABLE_OPTION$ -pe threaded 6
####
#### Request that the job only run on a node that has at least 15GB of RAM free.
#### If your analysis will require a lot of memory, set this option.
#REMOVE_THIS_STRING_TO_ENABLE_OPTION$ -l mem_free=15G 
#$ -l h_vmem=30G
#$ -l gpu
################################## END OF EMBEDDED SGE COMMANDS###################


# Trap the EXIT
trap 'echo exiting from ${0##*/}' SIGINT SIGKILL

# Check if Bash is loaded
##. /usr/share/Modules/init/bash || { echo "Failure to load /usr/share/Modules/init/bash";exit 1;}

__bashUtilities=/sbia/home/hsiehm/Scripts/createStandardScript/bash_utilities.sh
. ${__bashUtilities} || { echo "ERROR: Failed to source ${__bashUtilities} "; exit 1;}

################## Help file ####################
help()
{
cat <<HELP

Send deep learning pipeline.
 qsub -j y -o  /gpfs/cbica/home/santhosr/Codes/log.txt /gpfs/cbica/home/santhosr/Codes/runCode.sh

HELP
}

# Start of the program
# export CUDA_VISIBLE_DEVICES='' 
# exeFolder=`pwd`
# exeTime=`date`
# host=`hostname`
# echo "Command ${0##*/}"
# echo "Arguments $*"
# echo "Executing on ${host}"
# echo "Executing at ${exeTime}"
# echo "Executing in ${exeFolder}"

# Force python to flush the stdout/stderr
export PYTHONUNBUFFERED=TRUE

echo "Loading modules"
# module load python/anaconda/2
module unload cuda; module load cuda/8.0

export LD_LIBRARY_PATH=${CUDAHOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

nvidia-smi

echo "Running deep learning code"

# Path to main python file
python /gpfs/cbica/home/santhosr/Codes/jpegCreation.py




# python -W ignore jpegCreation.py  >> log.txt