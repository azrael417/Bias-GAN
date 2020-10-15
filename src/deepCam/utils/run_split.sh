#!/bin/bash



python3 distribute_files.py --input /raid1/data/tkurth/cam5_data/all_viz/hdf5 \
	--suffix ".h5" \
	--outputs \
	/raid1/data/tkurth/cam5_data/viz/hdf5/gpu0 \
	/raid1/data/tkurth/cam5_data/viz/hdf5/gpu1 \
	/raid2/data/tkurth/cam5_data/viz/hdf5/gpu2 \
	/raid2/data/tkurth/cam5_data/viz/hdf5/gpu3 \
	/raid3/data/tkurth/cam5_data/viz/hdf5/gpu4 \
        /raid3/data/tkurth/cam5_data/viz/hdf5/gpu5 \
        /raid4/data/tkurth/cam5_data/viz/hdf5/gpu6 \
        /raid4/data/tkurth/cam5_data/viz/hdf5/gpu7 \
	/raid5/data/tkurth/cam5_data/viz/hdf5/gpu8 \
        /raid5/data/tkurth/cam5_data/viz/hdf5/gpu9 \
        /raid6/data/tkurth/cam5_data/viz/hdf5/gpu10 \
        /raid6/data/tkurth/cam5_data/viz/hdf5/gpu11 \
        /raid7/data/tkurth/cam5_data/viz/hdf5/gpu12 \
        /raid7/data/tkurth/cam5_data/viz/hdf5/gpu13 \
        /raid8/data/tkurth/cam5_data/viz/hdf5/gpu14 \
        /raid8/data/tkurth/cam5_data/viz/hdf5/gpu15
