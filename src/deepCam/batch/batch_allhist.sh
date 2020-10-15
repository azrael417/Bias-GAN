#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=pytorch
#SBATCH --time=0-04:00:00
#SBATCH -C gpu
#SBATCH -c 80
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --output=batch_allhist.out
#SBATCH --error=batch_allhist.err
#SBATCH --mail-user=andregr@ethz.ch

module load python
module load pytorch/v1.2.0-gpu
srun python -u /global/homes/a/andregr/pytorch/Pytorch-DeepLab-v3-plus/train_allhist.py