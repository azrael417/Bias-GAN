#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib
export CUDA_VISIBLE_DEVICES=0

#total number of ranks
run_tag="infill_allnew_run2_1x1"
data_dir_prefix="/data/gpsro_data3_interp"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"
#checkpoint_file="${output_dir}/regressor_step_3200.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
cd ../gpsro_train
mpirun -np ${totalranks} ${mpioptions} ${profile} python train_infill.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "infill" \
       --noise_type "Normal" \
       --noise_dimensions 0 \
       --optimizer "AdamW" \
       --weight_decay 1e-2 \
       --start_lr 1e-3 \
       --lr_schedule type="multistep",milestones="80000",decay_rate="0.1" \
       --loss_weights valid=1.,hole=0.5,prc=0.,style=0.,tv=0. \
       --validation_frequency 200 \
       --training_visualization_frequency 100 \
       --validation_visualization_frequency 70 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 20000 \
       --disable_gds \
       --amp_opt_level "O1" \
       --local_batch_size 8 |& tee ${output_dir}/train.out
