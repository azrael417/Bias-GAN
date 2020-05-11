import os
import sys
import shutil
import numpy as np
import pandas as pd
import datetime as dt
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
files = [(os.path.dirname(x), os.path.basename(x).replace("data_in_","").replace(".npy","")) for x in files]

#put into dataframe
filesdf = pd.DataFrame(files, columns=["path", "timestamp"])
filesdf["date"] = filesdf["timestamp"].apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
validdf = filesdf[ (filesdf["date"].map(lambda x: x.year) == 2019) & (filesdf["date"].map(lambda x: x.day) <= 10) ]
testdf = filesdf[ (filesdf["date"].map(lambda x: x.year) == 2019) & (filesdf["date"].map(lambda x: x.day) >= 20) ]
traindf = filesdf[ ~ (filesdf["timestamp"].isin(validdf["timestamp"]) | filesdf["timestamp"].isin(testdf["timestamp"])) ]

#do the splitting
train_files = list(traindf.apply(lambda x: (x["path"], x["timestamp"]+'.npy'), axis=1))
validation_files = list(validdf.apply(lambda x: (x["path"], x["timestamp"]+'.npy'), axis=1))
test_files = list(testdf.apply(lambda x: (x["path"], x["timestamp"]+'.npy'), axis=1))

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
