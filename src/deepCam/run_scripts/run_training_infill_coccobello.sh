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
export CUDA_VISIBLE_DEVICES=1

#total number of ranks
config_name="infill3d_3.yaml"
run_tag="infill3d_dropout_test1"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#2D
#wandb agent "tkurth/GPSRO bias correction/9k2fob23"

#3D
#wandb agent "tkurth/GPSRO bias correction/ctdfauyf"

#3D new
#wandb agent "tkurth/GPSRO bias correction/mmysm2ms"

# manual runs
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/train_infill3d.py \
       --run_tag ${run_tag} \
       --config_file "../gpsro_configs/${config_name}"
