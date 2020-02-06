import os
import numpy as np
import netCDF4 as nc

nraid = 4

for idx in range(0,nraid):
    
    root = '/data{}/ecmwf_data'.format(2 * idx + 1)
    
    for dirr in os.listdirs(root):
        print(dirr)