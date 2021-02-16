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

OMP_NUM_THREADS=1 python main.py -m rank=0,1 gpus=2 experiment.name=coco_baseline dataset.tr_batch_size=40 dataset.num_workers=4 apex_opt=O2 metrics=mAP 
  yolo.lambda_xy: 0
  yolo.lambda_wh: 1.3615548670854327
  yolo.lambda_iou: 2.0955079100608427
  yolo.ignore_threshold: 0.4518141598777993
  yolo.lambda_conf: 1.409649659811485
  yolo.lambda_no_conf: 1.6585685019524925
  yolo.lambda_cls: 7.136283993679051
  yolo.iou_type: 3