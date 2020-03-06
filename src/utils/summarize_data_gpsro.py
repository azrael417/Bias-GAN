import os
import numpy as np

#merge function helper
def merge_token(token1, token2):
    #extract data
    #first
    n1 = token1[0]
    dmean1 = token1[1]
    dsqmean1 = token1[2]
    dmin1 = token1[3]
    dmax1 = token1[4]
    #second
    n2 = token2[0]
    dmean2 = token2[1]
    dsqmean2 = token2[2]
    dmin2 = token2[3]
    dmax2 = token2[4]

    #create new token
    nres = n1 + n2
    dmeanres = (n1/nres)*dmean1 + (n2/nres)*dmean2
    dsqmeanres = (n1/nres)*dsqmean1 + (n2/nres)*dsqmean2
    dminres = np.minimum(dmin1, dmin2)
    dmaxres = np.maximum(dmax1, dmax2)

    return (nres, dmeanres, dsqmeanres, dminres, dmaxres)


#create data token
def create_token(filename, data_format="nchw"):
    arr = np.load(filename).astype(np.float64)
    axis = (1,2) if data_format == "nchw" else (0,1)

    #how many samples do we have
    n = 1
    #compute stats
    mean = np.mean(arr, axis=axis)
    meansq = np.mean(np.square(arr), axis=axis)
    minimum = np.amin(arr, axis=axis)
    maximum = np.amax(arr, axis=axis)

    #result
    result = (n, mean, meansq, minimum, maximum)
    return result
        

#global parameters
nraid = 4
overwrite = False
data_format = "nchw"
data_path_prefix = "/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data/gpsro"

#init token
data_token = None
label_token = None

#root path
root = os.path.join( data_path_prefix, 'train' )
        
data_files = [ os.path.join(root, x)  for x in os.listdir(root) if x.endswith('.npy') and x.startswith('data_in_') ]
label_files = [ os.path.join(root, x)  for x in os.listdir(root) if x.endswith('.npy') and x.startswith('data_out_') ]

#get first token and then merge recursively
#data
data_token = create_token(data_files[0], data_format)
for filename in data_files[1:]:
    data_token = merge_token(create_token(filename, data_format), data_token)

#label
label_token = create_token(label_files[0], data_format)
for filename in label_files[1:]:
    label_token = merge_token(create_token(filename, data_format), label_token)

assert(data_token[0] == label_token[0])
    
#save token
np.savez(os.path.join(root, "stats.npz"), count=data_token[0],
         data_mean=data_token[1], data_sqmean=data_token[2], data_minval=data_token[3], data_maxval=data_token[4],
         label_mean=label_token[1], label_sqmean=label_token[2], label_minval=label_token[3], label_maxval=label_token[4])
