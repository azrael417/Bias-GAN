#!/bin/bash

#total ranks
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#parameters
run_tag="data2_adversarial_run4"
data_dir_prefix="/data1/gpsro_data3"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"
#checkpoint_file="${data_dir_prefix}/runs/gpsro_adversarial_run1-minmax/gan_step_3600.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
cd ../gpsro_train
mpirun -np ${totalranks} ${mpioptions} ${profile} python train_gan.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "gan" \
       --noise_type "Uniform" \
       --noise_dimensions 5 \
       --upsampler_type "Deconv1x" \
       --optimizer_generator "AdamW" \
       --optimizer_discriminator "AdamW" \
       --weight_decay 1e-3 \
       --start_lr_generator 1e-4 \
       --start_lr_discriminator 1e-4 \
       --update_frequency_generator 1 \
       --generator_warmup_steps 10000 \
       --update_frequency_discriminator 1 \
       --lr_schedule_generator type="multistep",milestones="30000",decay_rate="0.1" \
       --lr_schedule_discriminator type="multistep",milestones="30000",decay_rate="0.1" \
       --loss_type_gan "ModifiedMinMax" \
       --loss_weight_gan 1e-2 \
       --loss_weight_regression 1. \
       --enable_masks \
       --validation_frequency 100 \
       --training_visualization_frequency 100 \
       --validation_visualization_frequency 80 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 40000 \
       --disable_gds \
       --local_batch_size 4 |& tee ${output_dir}/train.out
