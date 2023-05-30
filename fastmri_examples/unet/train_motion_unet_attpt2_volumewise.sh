#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J fastMRI_pretrained
#SBATCH --output=/data2/motion_dnns/train_atmpt2_no_crop_nvol_240_nsub_2_volumewise/slurm_logs/slurm-%j.out
#SBATCH --error=/data2/motion_dnns/train_atmpt2_no_crop_nvol_240_nsub_2_volumewise/slurm_logs/slurm-%j.err
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end     
#SBATCH --mail-type=fail 
#SBATCH --mail-user=ainedineen@cusacklab.org


PYTHON="/opt/anaconda3/envs/fastmri/bin/python"
DATA="/data2/motion_dnns/motion_data"

${PYTHON} train_unet_motion_attpt2_volumewise_11_07_22.py --data_path ${DATA} \
    --num_workers 28
