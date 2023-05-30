#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH -J fastMRI_pretrained
#SBATCH --output=/data2/fastMRI/fastMRI/train_unet_brain_mc/slurm_logs_testing/slurm-%j.out
#SBATCH --error=/data2/fastMRI/fastMRI/train_unet_brain_mc/slurm_logs_testing/slurm-%j.err
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end     
#SBATCH --mail-type=fail 
#SBATCH --mail-user=ainedineen@cusacklab.org


PYTHON="/opt/anaconda3/envs/fastmri/bin/python"
DATA="/data2/fastMRI/fastMRI/data/multicoil_test"
OUTPUTS="/data2/fastMRI/fastMRI/train_unet_brain_mc/"
CHALLENGE='unet_brain_mc'

${PYTHON} run_trained_unet_inference.py --data_path ${DATA} \
    --output_path ${OUTPUTS} \
    --challenge ${CHALLENGE}

