#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#pick gpu
export CUDA_VISIBLE_DEVICES=0

#data dir
data_dir_prefix="/data/gpsro_data3_interp/test_2020"

#run 1
run_tag="regression3d_normal_comfy-sweep-4"
output_dir="${data_dir_prefix}/runs/${run_tag}/test_2020_step_18000"
checkpoint_file="/data/gpsro_data3_interp/runs/KEEP/${run_tag}/regressor3d_step_18000.cpt"

#run 2
run_tag="regression3d_normal_sage-sweep-4"
output_dir="${data_dir_prefix}/runs/${run_tag}/test_2020_step_12000"
checkpoint_file="/data/gpsro_data3_interp/runs/KEEP/${run_tag}/regressor_step_12000.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/eval_regression3d.py \
       --local_batch_size 5 \
       --checkpoint ${checkpoint_file} \
       --data_dir_prefix ${data_dir_prefix} \
       --data_set "test" \
       --output_dir ${output_dir} \
       --output_prefix "test_2020" \
       --model_prefix "regressor" \
       --layer_normalization "instance_norm" \
       --noise_type "Normal" \
       --noise_dimensions 0 \
       --enable_masks \
       --upsampler_type "Deconv1x" \
       --loss_type "l2" \
       --loss_weights valid=1.,hole=0.8 \
       --validation_visualization_frequency 10 \
       --disable_gds |& tee ${output_dir}/eval.out
