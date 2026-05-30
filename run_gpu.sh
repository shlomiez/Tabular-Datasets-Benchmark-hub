#!/bin/bash
#SBATCH --job-name=gpu_main_job
#SBATCH --output=gpu_main_job_%j.out
#SBATCH --error=gpu_main_job_%j.err
#SBATCH --partition=p_uriofir
#SBATCH --account=ug_uri_ofir
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=shlomie13@gmail.com
#SBATCH --mail-type=ALL

python main.py