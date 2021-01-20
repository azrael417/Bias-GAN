import sys
import os
import numpy as np
from time import sleep

import torch
from torch.utils.data import Dataset

#custom reader
import numpy_reader as nr


#dataset class
class GPSRODataset(Dataset):

    def init_files(self, source):
        self.source = source

        self.allfiles = sorted([ x.replace("data_in_", "") for x in os.listdir(self.source) if x.endswith(".npy") and x.startswith("data_in_") ])
        
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

        #parse the file sizes
        #data
        data_filename = os.path.join(self.source, "data_in_"+self.files[0])
        self.npr_data = nr.numpy_reader(split_axis = False, device = devindex)
        self.npr_data.num_intra_threads = num_intra_threads
        self.npr_data.parse(data_filename)
        #label
        label_filename = os.path.join(self.source, "data_out_"+self.files[0])
        self.npr_label = nr.numpy_reader(split_axis = False, device = devindex)
        self.npr_label.num_intra_threads = num_intra_threads
        self.npr_label.parse(label_filename)

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
        #store into tensors
        self.data_shift = torch.tensor(data_shift, requires_grad=False).to(self.read_device)
        self.data_scale = torch.tensor(data_scale, requires_grad=False).to(self.read_device)
        self.label_shift = torch.tensor(label_shift, requires_grad=False).to(self.read_device)
        self.label_scale = torch.tensor(label_scale, requires_grad=False).to(self.read_device)
        
        print("Initialized dataset with ", self.length, " samples.")

        
    def __len__(self):
        return self.length

    
    @property
    def shapes(self):
        return self.npr_data.shape, self.npr_label.shape

    
    def __getitem__(self, idx):
        #data
        data_file = os.path.realpath(os.path.join(self.source, "data_in_" + self.files[idx]))
        self.npr_data.init_file(data_file)
        data = self.npr_data.get_sample(0)
        self.npr_data.finalize_file()
        
        #label
        label_file = os.path.realpath(os.path.join(self.source, "data_out_" + self.files[idx]))
        self.npr_label.init_file(label_file)
        label = self.npr_label.get_sample(0)
        self.npr_label.finalize_file()
        
        #preprocess
        data[:, -1] = self.data_scale * (data[:, -1] - self.data_shift)
        label[:, -1] = self.label_scale * (label[:, -1] - self.label_shift)

        return data, label, self.files[idx]
