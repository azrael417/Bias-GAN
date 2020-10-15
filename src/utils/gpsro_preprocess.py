import os
import numpy as np

        
#global parameters
data_format = "nchw"
#data_path_prefix = "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/gpsro"
data_path_prefix = "/data/gpsro_data4_interp/train"
numfiles = 4

files = [ x.replace("data_in_", "") for x in os.listdir(os.path.join(data_path_prefix, "interpolated")) if x.endswith(".npy") and x.startswith("data_in_") ]

outdir = os.path.join(data_path_prefix, "all")
if not os.path.exists(outdir):
    os.makedirs(outdir)

# convert to nchw
for fname in files:
    #data
    data = np.load(os.path.join(data_path_prefix, "interpolated", "data_in_"+fname)).astype(np.float32)
    data = np.transpose(data, (2,0,1))
    np.save(os.path.join(outdir, "data_in_"+fname), data)

    #label
    label = np.load(os.path.join(data_path_prefix, "interpolated", "data_out_"+fname)).astype(np.float32)
    label = np.transpose(label, (2,0,1))
    np.save(os.path.join(outdir, "data_out_"+fname), label)

    
