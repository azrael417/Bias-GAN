import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#global parameters
nraid = 4
train_faction = 0.8
validation_fraction = 0.1
test_fraction = 0.1

for idx in range(0,nraid):

    #root path
    root = '/data{}/ecmwf_data'.format(2 * idx + 1)

    for gpudir in os.listdir(root):
        
        if (gpudir!="gpu0"):
            continue

        print(os.listdir(os.path.join(root,gpudir)))
