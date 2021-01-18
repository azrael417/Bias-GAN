import os
import numpy as np
from tqdm import tqdm

        
#global parameters
#data_format = "nchw"
#data_path_prefix = "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/gpsro"
#data_path_prefix = "/data/gpsro_data_hires/raw_interpolated"
#outdir = "/data/gpsro_data_hires/preproc_interpolated"
data_path_prefix = "/data/gpsro_data_hires/test_2020/raw_interpolated"
outdir = "/data/gpsro_data_hires/test_2020/preproc_interpolated"
#numfiles = 4

files = [ x.replace("data_in_", "") for x in os.listdir(data_path_prefix) if x.endswith(".npy") and x.startswith("data_in_") ]

# create output directory if not exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# convert to nchw
for fname in tqdm(files):
    #data
    data = np.load(os.path.join(data_path_prefix, "data_in_"+fname)).astype(np.float32)
    data = np.transpose(data, (2,0,1))
    np.save(os.path.join(outdir, "data_in_"+fname), data)

    #label
    label = np.load(os.path.join(data_path_prefix, "data_out_"+fname)).astype(np.float32)
    label = np.transpose(label, (2,0,1))
    np.save(os.path.join(outdir, "data_out_"+fname), label)

    
