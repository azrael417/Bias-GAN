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
run_tag="gpsro3_regression_final-shuffle-run-3"
data_dir_prefix="/data1/gpsro_data3_interp"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"
checkpoint_file="/data1/gpsro_data3_interp/runs/gpsro3_regression_run-1/regressor_step_11800.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} ${profile} python ../gpsro_train/train_regression.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "regressor" \
       --noise_dimensions 0 \
       --enable_masks \
       --upsampler_type "Deconv1x" \
       --optimizer "AdamW" \
       --weight_decay 1e-3 \
       --start_lr 1e-4 \
       --lr_schedule type="multistep",milestones="12000 40000",decay_rate="0.1" \
       --loss_weights valid=1.,hole=0.8 \
       --validation_frequency 100 \
       --training_visualization_frequency 200 \
       --validation_visualization_frequency 50 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 40000 \
       --disable_gds \
       --local_batch_size 4 |& tee ${output_dir}/train.out
