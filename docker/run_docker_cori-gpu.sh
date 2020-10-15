#!/bin/bash

#some parameters
#data_root=/home/cuda/tkurth/cam5_data

#inference runs
shifter --image registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest /bin/bash
	#--volume /global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data1:/data1 \
	#--volume /global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data3:/data3 \
	#--volume /global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data5:/data5 \
	#--volume /global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/data7:/data7 \
	#/bin/bash

#	--workdir "/opt/deepCam" /bin/bash
