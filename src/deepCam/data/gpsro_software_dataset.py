import os
import numpy as np
from time import sleep

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


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

        
    def __init__(self, source, statsfile, channels,
                 normalization_type = "MinMax", shuffle = True, masks = False,
                 shard_idx = 0, shard_num = 1,
                 num_intra_threads = 1, seed = 12345,
                 read_device = torch.device("cpu"), send_device = torch.device("cpu")):
        
        self.channels = channels
        self.normalization_type = normalization_type
        self.shuffle = shuffle
        self.masks = masks
        self.rng = np.random.RandomState(seed)
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.read_device = read_device
        self.send_device = send_device

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
        #label
        label_filename = os.path.join(self.source, "data_out_"+self.files[0])
        #masks
        if self.masks:
            masks_filename = os.path.join(self.source, "masks_"+self.files[0])
        
        #get shapes
        self.data_shape = np.load(data_filename).shape
        self.label_shape = np.load(label_filename).shape
        if self.masks:
            self.masks_shape = np.load(masks_filename).shape
        
        #set up the normalization
        statsfile = np.load(statsfile)

        #compute normalization
        if self.normalization_type == "MinMax":
            #get min and max:
            #data
            data_shift = statsfile["data_minval"][self.channels]
            data_scale = 1. / ( statsfile["data_maxval"][self.channels] - data_shift )
            #label
            label_shift = statsfile["label_minval"][self.channels]
            label_scale = 1. / ( statsfile["label_maxval"][self.channels] - label_shift )
        elif self.normalization_type == "MeanVariance":
            #get <x> and <x**2>:
            #data
            data_shift = statsfile["data_mean"][self.channels]
            data_scale = 1. / np.sqrt( statsfile["data_sqmean"][self.channels] - np.square(data_shift) )
            #label
            label_shift = statsfile["label_mean"][self.channels]
            label_scale = 1. / np.sqrt( statsfile["label_sqmean"][self.channels] - np.square(label_shift) )
            
        #reshape into broadcastable shape
        data_shift = np.reshape( data_shift, (data_shift.shape[0], 1, 1) ).astype(np.float32)
        data_scale = np.reshape( data_scale, (data_scale.shape[0], 1, 1) ).astype(np.float32)
        label_shift = np.reshape( label_shift, (label_shift.shape[0], 1, 1) ).astype(np.float32)
        label_scale = np.reshape( label_scale, (label_scale.shape[0], 1, 1) ).astype(np.float32)
        #store into tensors
        self.data_shift = torch.tensor(data_shift, requires_grad=False).to(self.send_device)
        self.data_scale = torch.tensor(data_scale, requires_grad=False).to(self.send_device)
        self.label_shift = torch.tensor(label_shift, requires_grad=False).to(self.send_device)
        self.label_scale = torch.tensor(label_scale, requires_grad=False).to(self.send_device)
        
        print("Initialized dataset with ", self.length, " samples.")

        
    def __len__(self):
        return self.length

    
    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    
    def __getitem__(self, idx):
        #data
        data_file = os.path.realpath(os.path.join(self.source, "data_in_" + self.files[idx]))
        data = torch.tensor(np.load(data_file)).to(self.send_device)

        #label
        label_file = os.path.realpath(os.path.join(self.source, "data_out_" + self.files[idx]))
        label = torch.tensor(np.load(label_file)).to(self.send_device)

        #mask
        if self.masks:
            mask_file = os.path.realpath(os.path.join(self.source, "masks_" + self.files[idx]))
            mask = torch.tensor(np.load(mask_file)).to(self.send_device)
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)
        label = self.label_scale * (label - self.label_shift)

        #result
        if self.masks:
            data, label, mask, self.files[idx]
        else:
            return data, label, self.files[idx]
        
