import os
import numpy as np

#merge function helper
def merge_token(token1, token2):
    #extract data
    #first
    n1 = token1[0]
    pmin1 = token1[1]
    dmean1 = token1[2]
    dsqmean1 = token1[3]
    dmin1 = token1[4]
    dmax1 = token1[5]
    #second
    n2 = token2[0]
    pmin2 = token2[1]
    dmean2 = token2[2]
    dsqmean2 = token2[3]
    dmin2 = token2[4]
    dmax2 = token2[5]

    #create new token
    nres = n1 + n2
    pminres = min([pmin1, pmin2])
    dmeanres = (n1/nres)*dmean1 + (n2/nres)*dmean2
    dsqmeanres = (n1/nres)*dsqmean1 + (n2/nres)*dsqmean2
    dminres = np.minimum(dmin1, dmin2)
    dmaxres = np.maximum(dmax1, dmax2)

    return (nres, pminres, dmeanres, dsqmeanres, dminres, dmaxres)


#create data token
def create_token(filename):
    #load data
    arr = np.load(filename).astype(np.float64)

    #how many samples do we have
    n = 1
    #compute stats
    point_minimum = arr.shape[0]
    data_mean = np.average(arr[:, 3:], axis=0)
    data_meansq = np.average(np.square(arr[:, 3:]), axis=0)
    data_minimum = np.amin(arr[:, 3:], axis=0)
    data_maximum = np.amax(arr[:, 3:], axis=0)

    #result
    result = (n, point_minimum, data_mean, data_meansq, data_minimum, data_maximum)
    return result
        

#global parameters
nraid = 4
overwrite = True
data_path_prefix = "/data/gpsro_data_hires/preproc_point"

#init token
data_token = None
label_token = None

#root path
root = data_path_prefix

data_files = sorted([ os.path.join(root, x)  for x in os.listdir(root) if x.endswith('.npy') and x.startswith('data_in_') ])
label_files = sorted([ os.path.join(root, x)  for x in os.listdir(root) if x.endswith('.npy') and x.startswith('data_out_') ])

#get first token and then merge recursively
data_token = create_token(data_files[0])
label_token = create_token(label_files[0])
for idx in range(1, len(data_files)):
    data_token = merge_token(create_token(data_files[idx]), data_token)
    label_token = merge_token(create_token(label_files[idx]), label_token)

assert(data_token[0] == label_token[0])
assert(data_token[1] == label_token[1])

# print stats
print("count: ", data_token[0])
print("count_minval: ", data_token[1])
print("data_mean: ", data_token[2])
print("data_sqmean: ", data_token[3])
print("data_minval: ", data_token[4])
print("data_maxval: ", data_token[5])
print("label_mean: ", label_token[2])
print("label_sqmean: ", label_token[3])
print("label_minval: ", label_token[4])
print("label_maxval: ", label_token[5])

#save token
np.savez(os.path.join(data_path_prefix, "stats_points.npz"), count=data_token[0], count_minval=data_token[1],
         data_mean=data_token[2], data_sqmean=data_token[3], data_minval=data_token[4], data_maxval=data_token[5],
         label_mean=label_token[2], label_sqmean=label_token[3], label_minval=label_token[4], label_maxval=label_token[5])
