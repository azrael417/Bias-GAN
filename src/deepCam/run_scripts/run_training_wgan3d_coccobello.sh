#!/bin/bash

#total ranks
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib
export CUDA_VISIBLE_DEVICES=0

#parameters
gpu_id=1
run_tag="adversarial3d_wgan_run3-2"
data_dir_prefix="/data/gpsro_data3_interp"
output_dir="${data_dir_prefix}/runs/${run_tag}"
checkpoint_file="${data_dir_prefix}/runs/adversarial3d_wgan_run3/gan_step_8000.cpt"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
cd ../gpsro_train
mpirun -np ${totalranks} ${mpioptions} python train_gan3d.py \
       --checkpoint ${checkpoint_file} \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --amp_opt_level "O1" \
       --model_prefix "gan" \
       --noise_type "Normal" \
       --noise_dimensions 3 \
       --upsampler_type "Deconv1x" \
       --optimizer_generator "AdamW" \
       --optimizer_discriminator "AdamW" \
       --weight_decay 0.05 \
       --start_lr_generator 0.00001 \
       --start_lr_discriminator 0.00001 \
       --generator_warmup_steps 0 \
       --relative_update_schedule type="static",update_frequency_generator=5,update_frequency_discriminator=1 \
       --lr_schedule_generator type="multistep",milestones="40000",decay_rate="0.1" \
       --lr_schedule_discriminator type="multistep",milestones="40000",decay_rate="0.1" \
       --loss_type_gan "Wasserstein" \
       --loss_weight_gan 1. \
       --loss_weight_regression 1. \
       --validation_frequency 100 \
       --training_visualization_frequency 100 \
       --validation_visualization_frequency 10 \
       --logging_frequency 50 \
       --save_frequency 200 \
       --max_steps 40000 \
       --disable_gds \
       --local_batch_size 8 |& tee ${output_dir}/train.out
