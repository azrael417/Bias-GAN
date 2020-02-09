#!/bin/bash

#some parameters
#data_root=/home/cuda/tkurth/cam5_data

#inference runs
shifterimg run \
	      --ipc host \
	      --volume "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data1:/data1:rw" \
	      --volume "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data3:/data3:rw" \
	      --volume "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data5:/data5:rw" \
	      --volume "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data7:/data7:rw" \
	      --workdir "/opt/deepCam" -it registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest /bin/bash
