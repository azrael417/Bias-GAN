#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=visualize
#SBATCH --time=0-04:00:00
#SBATCH -C gpu
#SBATCH -c 80
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --output=../../outputs/batch_visualize.out
#SBATCH --error=../../outputs/batch_visualize.err
#SBATCH --mail-user=lukaska@ethz.ch

conda activate pytorch-gpu
srun python -u ../test.py
