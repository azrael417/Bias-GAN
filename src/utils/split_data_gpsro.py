import os
import sys
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

#global parameters
train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1
bin_length = 10
seed = 13476251
#data_path_prefix = "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/gpsro"
data_path_prefix = "/data1/gpsro_data3_interp"
#data_path_prefix = "/data/gpsro_data"

#init rng
rng = np.random.RandomState(seed)

#go through files
outputpath = os.path.join(data_path_prefix)
        
#sort files to have definite state
files = sorted([ os.path.join(outputpath, 'all', x) for x in os.listdir(os.path.join(outputpath, 'all')) if x.endswith(".npy") and x.startswith("data_in_") ])
files = [(os.path.dirname(x), os.path.basename(x).replace("data_in_","")) for x in files]

#bin the data into bin_length
num_bins = len(files) // bin_length
files = files[:num_bins * bin_length]
files = np.array(files).reshape((num_bins, bin_length))

print(files)
sys.exit(1)

#shuffle files
rng.shuffle(files)

#take the train fraction
n_train = int(np.ceil(len(files)*train_fraction))
n_validation = int(np.ceil(len(files)*validation_fraction))
n_test = len(files) - n_validation - n_train
if n_test < 0:
    raise ValueError("Warning, you have more validations and training files than actual files in directory {}, adjust your splits.".format(os.path.join(outputpath, 'all')))

#do the splitting
train_files = files[:n_train]
validation_files = files[n_train:n_train+n_validation]
test_files = files[n_train+n_validation:]

#training files
#prepare directory
if os.path.isdir(os.path.join(outputpath,"train")):
    #clean up first
    shutil.rmtree(os.path.join(outputpath,"train"))
#create directory
os.makedirs(os.path.join(outputpath,"train"))
#loop over files and create symbolic links
for f in train_files:
            
    #data first
    infile = os.path.join(f[0],"data_in_"+f[1])
    outfile = os.path.join(outputpath,"train","data_in_"+f[1])
    os.symlink(infile, outfile)

    #label next
    infile = os.path.join(f[0],"data_out_"+f[1])
    outfile = os.path.join(outputpath,"train","data_out_"+f[1])
    os.symlink(infile, outfile)

    #mask
    infile = os.path.join(f[0],"masks_"+f[1])
    if os.path.exists(infile):
        outfile = os.path.join(outputpath,"train","masks_"+f[1])
        os.symlink(infile, outfile)
    
#validation files
#prepare directory
if os.path.isdir(os.path.join(outputpath,"validation")):
    #clean up first
    shutil.rmtree(os.path.join(outputpath,"validation"))
#create directory
os.makedirs(os.path.join(outputpath,"validation"))
#loop over files and create symbolic links
for f in validation_files:

    #data first
    infile = os.path.join(f[0],"data_in_"+f[1])
    outfile = os.path.join(outputpath,"validation","data_in_"+f[1])
    os.symlink(infile, outfile)

    #label next
    infile = os.path.join(f[0],"data_out_"+f[1])
    outfile = os.path.join(outputpath,"validation","data_out_"+f[1])
    os.symlink(infile, outfile)

    #mask
    infile = os.path.join(f[0],"masks_"+f[1])
    if os.path.exists(infile):
        outfile = os.path.join(outputpath,"validation","masks_"+f[1])
        os.symlink(infile, outfile)

#test files
#prepare directory
if os.path.isdir(os.path.join(outputpath,"test")):
    #clean up first
    shutil.rmtree(os.path.join(outputpath,"test"))
#create directory
os.makedirs(os.path.join(outputpath,"test"))
#loop over files and create symbolic links
for f in test_files:

    #data first
    infile = os.path.join(f[0],"data_in_"+f[1])
    outfile = os.path.join(outputpath,"test","data_in_"+f[1])
    os.symlink(infile, outfile)

    #label next
    infile = os.path.join(f[0],"data_out_"+f[1])
    outfile = os.path.join(outputpath,"test","data_out_"+f[1])
    os.symlink(infile, outfile)

    #mask
    infile = os.path.join(f[0],"masks_"+f[1])
    if os.path.exists(infile):
        outfile = os.path.join(outputpath,"test","masks_"+f[1])
        os.symlink(infile, outfile)
