#!/bin/bash

#some parameters
#data_root=/home/cuda/tkurth/cam5_data
data_root=/home/${USER}/tkurth/ecmwf_data
#data_root=/raid/data/tkurth/cam5_data
#data_root=/raid/data/tkurth

#test runs
#nvidia-docker run \
#	      --ipc host \
#              --env "CUDA_VISIBLE_DEVICES=0" \
#	      --workdir "/opt/numpy_reader/scripts" tkurth/pytorch-numpy_reader:latest ./reader_test.sh
#exit

#--device /dev/nvidia-fs0 \


#inference runs
nvidia-docker run \
	      --ipc host \
	      --env "CUDA_VISIBLE_DEVICES=0" \
	      --volume "${data_root}:/data:rw" \
	      --workdir "/opt/deepCam" -it tkurth/pytorch-bias_gan:latest /bin/bash
