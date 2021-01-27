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
run_tag="spherical_normal_debug"
config_name="spherical_hires.yaml"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#3D manual runs
mpirun -np ${totalranks} ${mpioptions} \
       python ../gpsro_train/train_spherical.py \
       --run_tag ${run_tag} \
       --tag_run \
       --config_file "../gpsro_configs/${config_name}"
