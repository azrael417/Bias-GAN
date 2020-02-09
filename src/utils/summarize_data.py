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
    axis = (0,2,3) if data_format == "nchw" else (0,1,2)

    #how many samples do we have
    n = arr.shape[0]
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
data_path_prefix = "/"

token = None
for idx in range(0,nraid):

    #root path
    root = os.path.join( data_path_prefix, 'data{}', 'ecmwf_data'.format(2 * idx + 1) )
    
    for gpudir in os.listdir(root):

        files = [ os.path.join(root, gpudir, 'train', x)  for x in os.listdir(os.path.join(root, gpudir, 'train')) \
                  if x.endswith('.npy') and x.startswith('data-') ]

        #get first token and then merge recursively
        token = create_token(files[0], data_format)
        for filename in files[1:]:
            token = merge_token(create_token(filename, data_format), token)


#distribute the file
for idx in range(0,nraid):

    #root path
    root = os.path.join( data_path_prefix, 'data{}', 'ecmwf_data'.format(2 * idx + 1) )

    for gpudir in os.listdir(root):
        
        #save results
        np.savez(os.path.join(root, gpudir, "train", "stats.npz"), count=token[0], mean=token[1], sqmean=token[2], minval=token[3], maxval=token[4])
