import sys
import os
import numpy as np
from time import sleep

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def gpsro_collate_fn(tensors):
    # radii are simple since there is not batch dependence on r:
    r_out = tensors[0][0]

    # determine max padding value
    max_length = max([t[1].shape[0] for t in tensors])
    
    # pad the stuff
    phi_out = torch.stack([F.pad(t[1], (0, max_length - t[1].shape[0])) for t in tensors], axis=0)
    theta_out = torch.stack([F.pad(t[2], (0, max_length - t[2].shape[0])) for t in tensors], axis=0)
    area_out = torch.stack([F.pad(t[3], (0, max_length - t[3].shape[0])) for t in tensors], axis=0)
    data_out = torch.stack([F.pad(t[4], (0, max_length - t[4].shape[0], 0, 0)) for t in tensors], axis=0)
    label_out = torch.stack([F.pad(t[5], (0, max_length - t[5].shape[0], 0, 0)) for t in tensors], axis=0)
    files_out = torch.stack([t[6] for t in tensors], axis=0)

    return r_out, phi_out, theta_out, area_out, data_out, label_out, files_out

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
        data_shift = np.expand_dims(data_shift.astype(np.float32), axis=0)
        data_scale = np.expand_dims(data_scale.astype(np.float32), axis=0)
        label_shift = np.expand_dims(label_shift.astype(np.float32), axis=0)
        label_scale = np.expand_dims(label_scale.astype(np.float32), axis=0)
        
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
