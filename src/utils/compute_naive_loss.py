import os
import numpy as np

#global parameters
nraid = 4
overwrite = False
data_format = "nchw"
data_path_prefix = "/"
channels = [0, 1, 2, 3, 4, 9]

#start loop
loss = 0.
count = 0
for idx in range(0,nraid):

    #root path
    root = os.path.join( data_path_prefix, 'data{}'.format(2 * idx + 1), 'ecmwf_data' )
    
    for gpudir in os.listdir(root):

        if not gpudir.startswith("gpu"):
            continue

        #load normalization
        normarr = np.load(os.path.join(root, gpudir, 'train', 'stats.npz'))
        minarr = normarr["minval"].reshape(1, normarr["minval"].shape[0], 1, 1)
        minarr = minarr[:,channels,...]
        maxarr = normarr["maxval"].reshape(1, normarr["maxval"].shape[0], 1, 1)
        maxarr = maxarr[:,channels,...]

        #go through files in valid, normalize and compute loss
        files = [ os.path.join(root, gpudir, 'validation', x)  for x in os.listdir(os.path.join(root, gpudir, 'validation')) \
                  if x.endswith('.npy') and x.startswith('data-') ]

        for filename in files:
            #load data and project
            token = np.load(filename)
            token = token[:, channels, ...]
            #increase counter
            current_count = token.shape[0]
            count += current_count
            #normalize
            token = (token - minarr) / (maxarr - minarr)
            #split in data and label
            data = token[0::2]
            label = token[1::2]
            #compute diff
            absdiff = np.abs(data-label)
            current_loss = np.sum(np.mean(absdiff, axis=(1,2,3)))
            loss += current_loss
            #print temp loss
            print("Average loss for file {}: {}".format(filename, current_loss / float(current_count)))
            
#print result
print("Final average loss: {}", loss / float(count))
