#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -J ecmwf_train ing
#SBATCH -A m1759
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=30GB
#SBATCH -t 04:10:00
#SBATCH --image=registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest
#SBATCH --volume="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data1:/data1:rw"
#SBATCH --volume="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data3:/data3:rw"
#SBATCH --volume="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data5:/data5:rw"
#SBATCH --volume="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data7:/data7:rw"

