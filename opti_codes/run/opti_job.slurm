#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --job-name=Pteropods
#SBATCH --output=/home/salbernhe/workspace/Seapopym-optimization/opti_codes/run/%j.out

source /home/salbernhe/miniforge3/etc/profile.d/conda.sh && \
conda activate seapopym_optimization_env && \
python /home/salbernhe/workspace/Seapopym-optimization/opti_codes/Scripts/pteropods_papa.py