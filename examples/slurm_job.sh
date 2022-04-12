#!/bin/bash           
#
#SBATCH --job-name=ig_navigation_test
#SBATCH --partition=viscam
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

source ~/.bashrc  
conda activate ig_navigation

export GIBSON_ASSETS_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/assets
export IGIBSON_DATASET_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/ig_dataset

cd /sailhome/mjlbach/Repositories/ig_navigation

python train.py ++experiment_name=$SLURM_JOB_NAME +experiment=search
