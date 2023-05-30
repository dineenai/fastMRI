#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH -J test_motion_net
#SBATCH --output=/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/slurm_logs_testing/slurm-%j.out
#SBATCH --error=/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/slurm_logs_testing/slurm-%j.err
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end     
#SBATCH --mail-type=fail 
#SBATCH --mail-user=ainedineen@cusacklab.org


PYTHON="/opt/anaconda3/envs/fastmri/bin/python"
DATA="/data2/motion_dnns/motion_data/train"
OUTPUTS="/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/"

${PYTHON} run_unet_inference_train_atmpt1_single_vol_no_crop_n240_test.py --data_path ${DATA} \
    --output_path ${OUTPUTS}

