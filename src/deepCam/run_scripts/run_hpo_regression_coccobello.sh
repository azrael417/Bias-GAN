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
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

#total number of ranks
config_name="regression_hires_hpo.yaml"
group_tag="hires_hpo_regression"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#run the stuff
#2D minmax normalized smooth-l1 sweep
#wandb agent "tkurth/GPSRO bias correction/j9qwx820"

#2D mean-variance normalized smooth-l1 sweep
#wandb agent "tkurth/GPSRO bias correction/vk4bexny"

#2D mean-variance normalized l2 sweep
#wandb agent "tkurth/GPSRO bias correction/vj9nbuny"

#3D minmax normalized smooth-l1 sweep
#wandb agent "tkurth/GPSRO bias correction/kaby4dny"

#3D mean-variance normalized smooth-l1 sweep
#wandb agent "tkurth/GPSRO bias correction/wz9toagz"

#3D mean-variance normalized l2 sweep 
#wandb agent "tkurth/GPSRO bias correction/s7v3phxp"

#3D mean-variance normalized l2 sweep second gen
#wandb agent "tkurth/GPSRO bias correction/1kv1bt8n"

#3D mean-variance normalized smooth-l1 cosine annealing sweep
#wandb agent "tkurth/GPSRO bias correction/wtodmbsu"

#3D mean-variance normalized l2 cosine annealing sweep
#wandb agent "tkurth/GPSRO bias correction/lcphhj62"

# 3D manual run
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/hpo_regression3d.py \
       --group_tag ${group_tag} \
       --config_file "../gpsro_configs/${config_name}"
