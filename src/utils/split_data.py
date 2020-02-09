import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

#global parameters
nraid = 4
train_fraction = 0.6
validation_fraction = 0.2
test_fraction = 0.2
seed = 13476251

#init rng
rng = np.random.RandomState(seed)

#go through files
for idx in range(0,nraid):

    #root path
    root = '/data{}/ecmwf_data'.format(2 * idx + 1)

    for gpudir in os.listdir(root):

        if not gpudir.startswith("gpu"):
            continue
        
        #if (gpudir!="gpu0"):
        #    continue

        #this will be the outputpath
        outputpath = os.path.join(root, gpudir)
        
        #sor files to have definite state
        files = sorted([ os.path.join(outputpath, 'all', x) for x in os.listdir(os.path.join(outputpath, 'all')) if x.endswith(".npy") and x.startswith("data-") ])
        files = [(os.path.dirname(x), os.path.basename(x).replace("data-","")) for x in files]

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
            infile = os.path.join(f[0],"data-"+f[1])
            outfile = os.path.join(outputpath,"train","data-"+f[1])
            os.symlink(infile, outfile)
            
            #filenames next
            infile = os.path.join(f[0],"filenames-"+f[1])
            outfile = os.path.join(outputpath,"train","filenames-"+f[1])
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
            infile = os.path.join(f[0],"data-"+f[1])
            outfile = os.path.join(outputpath,"validation","data-"+f[1])
            os.symlink(infile, outfile)

            #filenames next
            infile = os.path.join(f[0],"filenames-"+f[1])
            outfile = os.path.join(outputpath,"validation","filenames-"+f[1])
            os.symlink(infile, outfile)

