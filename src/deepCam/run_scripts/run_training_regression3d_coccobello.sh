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
#run_tag="regression3d_comfy-sweep-4"
#data_dir_prefix="/data/gpsro_data3_interp"
run_tag="regression3d_sage-sweep-4"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"
#checkpoint_file="/data1/gpsro_data3_interp/runs/gpsro3_regression_run-1/regressor_step_11800.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#config
config=../configs/sage-sweep-4.yaml

#prepare dir:
mkdir -p ${output_dir}

#new stuff
mpirun -np ${totalranks} ${mpioptions} ${profile} python ../gpsro_train/hpo_regression3d.py \
       --config_file ${config}

exit

#run the stuff
mpirun -np ${totalranks} ${mpioptions} ${profile} python ../gpsro_train/train_regression3d.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "regressor" \
       --noise_type "Normal" \
       --noise_dimensions 0 \
       --enable_masks \
       --upsampler_type "Deconv1x" \
       --optimizer "AdamW" \
       --adam_eps 0.0008632 \
       --layer_normalization "instance_norm" \
       --weight_decay 0.1199 \
       --start_lr 0.003743 \
       --lr_schedule type="multistep",milestones="10000",decay_rate="0.0336" \
       --loss_type "l2" \
       --loss_weights valid=1.,hole=0.8 \
       --validation_frequency 100 \
       --training_visualization_frequency 200 \
       --validation_visualization_frequency 50 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 11000 \
       --disable_gds \
       --enable_amp \
       --local_batch_size 7 |& tee ${output_dir}/train.out
