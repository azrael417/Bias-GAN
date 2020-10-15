#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    echo "Please pass a number of ranks to this bash script"
    exit
fi

#env
export OMPI_MCA_btl=^openib

#total number of ranks
totalranks=$1
run_tag="gpsro_gan_run1"
data_dir_prefix="/data/gpsro_data"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"
checkpoint_file="${output_dir}/regressor_step_4800.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} ${profile} python train_gan_gpsro.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "gan" \
       --noise_dimensions 0 \
       --upsampler_type "Deconv1x" \
       --optimizer "AdamW" \
       --weight_decay 1e-5 \
       --start_lr_generator 1e-4 \
       --start_lr_discriminator 1e-4 \
       --update_frequency_generator 1 \
       --update_frequency_discriminator 1 \
       --lr_schedule_generator type="multistep",milestones="4000 6000",decay_rate="0.1" \
       --lr_schedule_discriminator type="multistep",milestones="4000 6000",decay_rate="0.1" \
       --validation_frequency 100 \
       --training_visualization_frequency 100 \
       --validation_visualization_frequency 10 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 40000 \
       --disable_gds \
       --local_batch_size 4 |& tee ${output_dir}/train.out
