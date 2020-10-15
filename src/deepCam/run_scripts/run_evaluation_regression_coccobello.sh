#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#data dir
data_dir_prefix="/data/gpsro_data3_interp/test_2020"

#run 1
run_tag="regression_normal_jolly-sweep-7"
output_dir="${data_dir_prefix}/runs/${run_tag}/test_2020_step_11800"
checkpoint_file="/data/gpsro_data3_interp/runs/KEEP/regression_normal_jolly-sweep-7/regressor_step_11800.cpt"
normalization="Normal"

#run 2
#run_tag="regression_uniform_upbeat-sweep-13"
#output_dir="${data_dir_prefix}/runs/${run_tag}/test_2020_step_15000"
#checkpoint_file="/data/gpsro_data3_interp/runs/KEEP/regression_uniform_upbeat-sweep-13/regressor_step_15000.cpt"
#normalization="Uniform"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} python ../gpsro_train/eval_regression.py \
       --checkpoint ${checkpoint_file} \
       --data_dir_prefix ${data_dir_prefix} \
       --data_set "test" \
       --output_dir ${output_dir} \
       --output_prefix "test_2020" \
       --model_prefix "regressor" \
       --noise_type ${normalization} \
       --noise_dimensions 0 \
       --enable_masks \
       --upsampler_type "Deconv1x" \
       --loss_type "smooth_l1" \
       --loss_weights valid=1.,hole=0.8 \
       --validation_visualization_frequency 10 \
       --disable_gds |& tee ${output_dir}/eval.out
