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
export CUDA_VISIBLE_DEVICES=0

#total number of ranks
run_tag="regression3d_normal_comfy-sweep-4"
config_name="good/regression3d_normal_comfy-sweep-4.yaml"

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
wandb agent "tkurth/GPSRO bias correction/lcphhj62"

#2D manual runs
#mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/hpo_regression.py \
#       --run_tag ${run_tag} \
#       --tag_run \
#       --config_file "../gpsro_configs/${config_name}"

##3D manual runs
#mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/hpo_regression3d.py \
#       --run_tag ${run_tag} \
#       --tag_run \
#       --config_file "../gpsro_configs/${config_name}"
