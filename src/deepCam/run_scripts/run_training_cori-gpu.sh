#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -J ecmwf_training
#SBATCH -A m1759
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=30GB
#SBATCH -t 03:59:00
#SBATCH --image=registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest

#env
export OMPI_MCA_btl=^openib

#total number of ranks
rankspernode=8
totalranks=$(( ${rankspernode} * ${SLURM_NNODES} ))
#run_tag="era_prediction_clean3_run6-gauss"
run_tag="gpsro_prediction_run1"
data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/gpsro"
#data_dir_prefix="/global/cscratch1/sd/tkurth/ECMWF/data"
output_dir="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/runs/gpsro/${run_tag}"
#checkpoint_file="${data_dir_prefix}/data1/ecmwf_data/era_prediction_clean3_run5-1-gauss/regressor_step_7600.cpt"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
srun -N ${SLURM_NNODES} -n ${totalranks} -c $(( 40 / ${rankspernode} )) --cpu_bind=cores -u \
     shifter --image=registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest \
     python train_regression_gpsro.py \
     --run_tag ${run_tag} \
     --data_dir_prefix "${data_dir_prefix}" \
     --output_dir "${output_dir}" \
     --model_prefix "regressor" \
     --noise_dimensions 2 \
     --noise_type "Normal" \
     --weight_decay 1e-3 \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="4000 8000 12000",decay_rate="0.1" \
     --validation_frequency 200 \
     --visualization_frequency 200 \
     --logging_frequency 50 \
     --save_frequency 400 \
     --max_steps 50000 \
     --disable_gds \
     --local_batch_size 2 |& tee ${output_dir}/train.out
