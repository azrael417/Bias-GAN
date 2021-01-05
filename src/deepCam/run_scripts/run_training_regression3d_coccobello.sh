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

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#config
config=../gpsro_configs/good/regression3d_hires_gentle-dawn-703.yaml
group_tag="hires_hpo_regression"
run_tag="regression3d_hires_gentle-dawn-703"

#config=../gpsro_configs/regression_hires.yaml
#group_tag="hires_hpo_regression"

#prepare dir:
mkdir -p ${output_dir}

#new stuff
mpirun -np ${totalranks} ${mpioptions} ${profile} python ../gpsro_train/train_regression3d.py \
       --run_tag ${run_tag} \
       --group_tag ${group_tag} \
       --config_file ${config}
