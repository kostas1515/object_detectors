#!/bin/bash -l
# Use the current working directory
#SBATCH -D ../slurm
# Reset environment for this job.
# SBATCH --export=NONE
#Specify the GPU partition
#SBATCH -p gpu,gpuc
#Specify the number of GPUs to be used
#SBATCH --gres=gpu:2
# Define job name
#SBATCH -J COCO
# Alocate memeory per core
#SBATCH --mem-per-cpu=32000M
# Setting maximum time days-hh:mm:ss]
#SBATCH -t 72:00:00
# Setting number of CPU cores and number of nodes
#SBATCH -n 4 -N 1

# Load modules
module load libs/nvidia-cuda/10.1.168/bin

# Change conda env
conda activate dds

cd ..

python test.py -m gpus=2 experiment.name=lvis_baseline dataset=lvis dataset.tr_batch_size=24 apex_opt=O2