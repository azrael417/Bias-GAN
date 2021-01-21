import sys
import os
import numpy as np
from time import sleep

import torch
from torch.utils.data import Dataset


#dataset class
class GPSRODataset(Dataset):

    def init_files(self, source):
        self.source = source

        self.allfiles = sorted([ x for x in os.listdir(self.source) if x.endswith(".npz") ])
        
        if self.shuffle:
            self.rng.shuffle(self.allfiles)

        #shard the dataset
        shard_size = len(self.allfiles) // self.shard_num
        start = shard_size * self.shard_idx
        end = start + shard_size
        self.files = self.allfiles[start:end]
            
        self.length = len(self.files)

        
    def __init__(self, source, statsfile,
                 normalization_type = "MinMax", shuffle = True
                 shard_idx = 0, shard_num = 1,
                 num_intra_threads = 1, seed = 12345,
                 read_device = torch.device("cpu")):
        
        self.normalization_type = normalization_type
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.read_device = read_device

        #set up files
        self.init_files(source)

        #set up reader
        if (self.read_device == torch.device("cpu")):
            devindex = -1
        else:
            devindex = self.read_device.index

        #set up the normalization
        statsfile = np.load(statsfile)

        #compute normalization
        if self.normalization_type == "MinMax":
            #get min and max:
            #data
            data_shift = statsfile["data_minval"]
            data_scale = 1. / ( statsfile["data_maxval"] - data_shift )
            #label
            label_shift = statsfile["label_minval"]
            label_scale = 1. / ( statsfile["label_maxval"] - label_shift )
        elif self.normalization_type == "MeanVariance":
            #get <x> and <x**2>:
            #data
            data_shift = statsfile["data_mean"]
            data_scale = 1. / np.sqrt( statsfile["data_sqmean"] - np.square(data_shift) )
            #label
            label_shift = statsfile["label_mean"]
            label_scale = 1. / np.sqrt( statsfile["label_sqmean"] - np.square(label_shift) )
            
        #reshape into broadcastable shape
        data_shift = data_shift.astype(np.float32)
        data_scale = data_scale.astype(np.float32)
        label_shift = label_shift.astype(np.float32)
        label_scale = label_scale.astype(np.float32)
        
        print("Initialized dataset with ", self.length, " samples.")

        
    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):
        # open file
        infile = os.path.realpath(os.path.join(self.source, self.files[idx]))
        token = numpy.load(infile)
        
        #coords
        r = token['r']
        phi = token['phi']
        theta = token['theta']
        area = token['area']
        
        #data
        data = token['data']
        
        #label
	label = token['label']
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)
        label = self.label_scale * (label - self.label_shift)

        return (r, phi, theta, area, data, label, self.files[idx])
