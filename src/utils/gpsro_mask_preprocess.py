import os
import re
import numpy as np
from scipy.interpolate import SmoothBivariateSpline as interp
from scipy.sparse import coo_matrix
from tqdm import tqdm

file_root = "/data/gpsro_data_hires/raw"
output_dir = "/data/gpsro_data_hires/preproc_mask"
pattern = re.compile("data_in_raw_(.*?).data")
tags = {pattern.match(x).groups()[0] for x in os.listdir(file_root) if pattern.match(x) is not None}

# create directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# iterate over tags
for tag in tqdm(tags):
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    lon = [ ((x + 180) / 10).astype(np.int) for x in lon]
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')
    lat = [ ((x + 90) / 10).astype(np.int) for x in lat]
    # load data in
    data_in = np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data out
    data_out = np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    
    # create sparse matrix per channel
    data_in_mat = [ coo_matrix( (x[0], (x[1], x[2])), shape=(19, 37) ) for x in zip(data_in, lat, lon) ]
    data_out_mat = [ coo_matrix( (x[0], (x[1], x[2])), shape=(19, 37) ) for x in zip(data_out, lat, lon) ]
    mask_mat = [ coo_matrix( (np.ones(len(x[0]), dtype=np.float32), (x[0], x[1])), shape=(19, 37) ) for x in zip(lat, lon) ]

    # create dense arrays
    data_in_arr = np.stack([x.astype(np.float32).toarray() for x in data_in_mat], axis=0)
    data_out_arr = np.stack([x.astype(np.float32).toarray() for x in data_out_mat], axis=0)
    mask_arr = np.stack([x.toarray() for x in mask_mat], axis=0)
    
    # store results
    np.save(os.path.join(output_dir, "data_in_" + tag + ".npy"), data_in_arr)
    np.save(os.path.join(output_dir, "data_out_" + tag + ".npy"), data_out_arr)
    np.save(os.path.join(output_dir, "masks_" + tag + ".npy"), mask_arr)
