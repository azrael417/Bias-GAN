import os
import shutil as sh
import numpy as np

#global parameters
nraid = 4
overwrite = False
data_format = "nchw"
#input_file_path = "/"
input_file_path = "/global/cscratch1/sd/tkurth/ECMWF/data"
#data_path_prefix = "/"
data_path_prefix = "/global/cscratch1/sd/tkurth/ECMWF/data"

#create inputfilename
input_file = os.path.join(input_file_path, "stats.npz")

#distribute the file
for idx in range(0,nraid):

    #root path
    root = os.path.join( data_path_prefix, 'data{}'.format(2 * idx + 1), 'ecmwf_data' )

    for gpudir in os.listdir(root):

        if not gpudir.startswith("gpu"):
            continue
        
        #output file
        output_file = os.path.join(root, gpudir, "train", "stats.npz")
        
        #copy
        sh.copyfile(input_file, output_file)
