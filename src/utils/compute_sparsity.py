import os
import numpy as np

data_path = "/data/gpsro_data3_interp/all"

maskfiles = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("masks_") and x.endswith(".npy")]

sparsity = 0.
norm = 1. / float(len(maskfiles))
for fname in maskfiles:
    mask = np.load(fname)
    sparsity += (np.sum(mask) / np.prod(mask.shape)) * norm

print("Average sparsity: {:2.1f}%".format(100.*sparsity))
