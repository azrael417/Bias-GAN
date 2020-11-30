#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#set the devices
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8,9,10,11,12,13,14,15

#total number of ranks
config_name="infill3d_hires_hpo.yaml"
group_tag="hires_hpo"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#2D
#wandb agent "tkurth/GPSRO bias correction/9k2fob23"

#3D
#wandb agent "tkurth/GPSRO bias correction/ctdfauyf"

#3D new
#wandb agent "tkurth/GPSRO bias correction/mmysm2ms"

# manual runs
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/hpo_infill3d.py \
       --group_tag ${group_tag} \
       --config_file "../gpsro_configs/${config_name}"
