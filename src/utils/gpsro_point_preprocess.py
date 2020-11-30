import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline as interp
from scipy.sparse import coo_matrix
from tqdm import tqdm

file_root = "/data/gpsro_data_hires/raw"
level_file = "./gpsro_metadata.csv"
output_dir = "/data/gpsro_data_hires/preproc_point"
pattern = re.compile("data_in_raw_(.*?).data")
tags = {pattern.match(x).groups()[0] for x in os.listdir(file_root) if pattern.match(x) is not None}
earth_radius = 6371000

# special settings
matrix_shape = (91, 181) # 2 degree resolution
degree = 2.
levels = list(range(20,65))
altitude_tag = "geometric_altitude"

# create directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# open level file and select
metadf = pd.read_csv(level_file)
metadf = metadf[ metadf["level"].isin(levels) ].sort_values(by="level", ascending = True).reset_index(drop = True)
    
# iterate over tags
for tag in tqdm(tags):
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data in
    data_in = np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data out
    data_out = np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')

    # 
    print(lon, data_in)
    sys.exit(1)
    
    # store results
    np.save(os.path.join(output_dir, "data_in_" + tag + ".npy"), data_in_arr)
    np.save(os.path.join(output_dir, "data_out_" + tag + ".npy"), data_out_arr)
    np.save(os.path.join(output_dir, "masks_" + tag + ".npy"), mask_arr)
