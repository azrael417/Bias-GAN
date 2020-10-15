#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#total number of ranks
run_tag="regression_upbeat-sweep-13"
data_dir_prefix="/data1/gpsro_data3_interp"
output_dir="${data_dir_prefix}/runs/${run_tag}"
checkpoint_file="/data1/gpsro_data3_interp/runs/${run_tag}/regressor_step_15000.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/eval_regression.py \
       --data_dir_prefix ${data_dir_prefix} \
       --data_set "validation" \
       --output_dir ${output_dir} \
       --output_prefix "validation" \
       --model_prefix "regressor" \
       --noise_type "Uniform" \
       --noise_dimensions 0 \
       --enable_masks \
       --upsampler_type "Deconv1x" \
       --loss_type "smooth_l1" \
       --loss_weights valid=1.,hole=0.8 \
       --validation_visualization_frequency 10 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 40000 \
       --disable_gds |& tee ${output_dir}/eval.out
