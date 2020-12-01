import os
import re
import itertools
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
altitudes = [ (earth_radius + x) / earth_radius for x in metadf[ altitude_tag ].values.tolist() ]
    
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

    # compute spherical coordinates
    phi = [ np.pi * (x + 180.) / 180. for x in lon ]
    theta = [ np.pi * (90. - x) / 180. for x in lat ]

    # compute x, y, z:
    xcoords = [ a * np.cos(p) * np.sin(t) for a, p, t in zip(altitudes, phi, theta) ]
    ycoords = [ a * np.sin(p) * np.sin(t) for a, p, t in zip(altitudes, phi, theta) ]
    zcoords = [ a * np.cos(t) for a, p, t in zip(altitudes, phi, theta) ]

    # flatten
    xcoords = np.array(list(itertools.chain(*[ x.tolist() for x in xcoords ])), dtype=np.float32)
    ycoords = np.array(list(itertools.chain(*[ y.tolist() for y in ycoords ])), dtype=np.float32)
    zcoords = np.array(list(itertools.chain(*[ z.tolist() for z in zcoords ])), dtype=np.float32)
    data_in = np.array(list(itertools.chain(*[ d.tolist() for d in data_in ])), dtype=np.float32)
    data_out = np.array(list(itertools.chain(*[ d.tolist() for d in data_out ])), dtype=np.float32)

    # compute final arrays
    data_in = np.stack([xcoords, ycoords, zcoords, data_in], axis = 1)
    data_out = np.stack([xcoords, ycoords, zcoords, data_out], axis = 1)
    
    # store results
    np.save(os.path.join(output_dir, "data_in_" + tag + ".npy"), data_in)
    np.save(os.path.join(output_dir, "data_out_" + tag + ".npy"), data_out)
