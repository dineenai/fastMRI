#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J fastMRI_pretrained
#SBATCH --output=/data2/fastMRI/fastMRI/train_unet_brain_mc/slurm_logs/slurm-%j.out
#SBATCH --error=/data2/fastMRI/fastMRI/train_unet_brain_mc/slurm_logs/slurm-%j.err
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end     
#SBATCH --mail-type=fail 
#SBATCH --mail-user=ainedineen@cusacklab.org


PYTHON="/opt/anaconda3/envs/fastmri/bin/python"
DATA="/data2/fastMRI/fastMRI/data/"
CHALLENGE='multicoil'
MASK_TYPE='random'
ROOT='/data2/fastMRI/fastMRI/train_unet_brain_mc'

${PYTHON} train_unet_demo_cust_dir.py --data_path ${DATA} \
    --challenge ${CHALLENGE} \
    --mask_type ${MASK_TYPE} \
    --num_workers 28
