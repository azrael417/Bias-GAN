import sys
import os
import numpy as np
from time import sleep

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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
        self.npr_data = nr.numpy_reader(split_axis = False, device = devindex)
        self.npr_data.num_intra_threads = num_intra_threads
        self.npr_data.parse(data_filename)
        #label
        label_filename = os.path.join(self.source, "data_out_"+self.files[0])
        self.npr_label = nr.numpy_reader(split_axis = False, device = devindex)
        self.npr_label.num_intra_threads = num_intra_threads
        self.npr_label.parse(label_filename)
        #masks
        if self.masks is not None:
            mask_filename = os.path.join(self.source, "masks_"+self.files[0])
            self.npr_mask = nr.numpy_reader(split_axis = False, device = devindex)
            self.npr_mask.num_intra_threads = num_intra_threads
            self.npr_mask.parse(mask_filename)

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

        #masks
        if self.masks is not None:
            mask_file = os.path.realpath(os.path.join(self.source, "masks_" + self.files[idx]))
            self.npr_mask.init_file(mask_file)
            mask = self.npr_mask.get_sample(0)
            self.npr_mask.finalize_file()
        
        #upload if necessary
        if (data.device != self.send_device):
            data = data.to(self.send_device)
        if (label.device != self.send_device):
            label = label.to(self.send_device)
        if self.masks and (mask.device != self.send_device):
            mask = mask.to(self.send_device)

        #preprocess
        data = self.data_scale * (data - self.data_shift)
        label = self.label_scale * (label - self.label_shift)

        if self.masks:
            return data, label, mask, self.files[idx]
        else:
            return data, label, self.files[idx]
