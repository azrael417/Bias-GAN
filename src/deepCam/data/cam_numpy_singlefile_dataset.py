import os
import h5py as h5
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
class CamDataset(Dataset):
  
    def __init__(self, filename_filenames, filename_data, filename_label, channels, device = -1, preprocess = True):
        #channels
        self.channels = channels

        #preprocessing?
        self.preprocess = preprocess

        #read filenames first
        self.files = list(np.load(filename_filenames))
        
        #init numpy loader
        #create loaders and parse header
        #data
        self.npr_data = nr.numpy_reader(split_axis = True, device = device)
        self.npr_data.parse(filename_data)
        self.npr_data.init_file(filename_data)

        #label
        self.npr_label = nr.numpy_reader(split_axis = True, device = device)
        self.npr_label.parse(filename_label)
        self.npr_label.init_file(filename_label)

        #label
        self.length = self.npr_data.num_samples
        assert(self.length == self.npr_label.num_samples)
        assert(self.length == len(self.files))
        
        print("Initialized dataset with ", self.length, " samples.")

    def __del__(self):
        self.npr_data.finalize_file()
        self.npr_label.finalize_file()
        
    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.npr_data.shape, self.npr_label.shape
    
    def __getitem__(self, idx):
        
        #load data
        try:
            X = self.npr_data.get_sample(idx)
            if self.preprocess:
                X = X[..., self.channels].permute(2, 0, 1)

            Y = self.npr_label.get_sample(idx)
            if self.preprocess:
                Y = torch.squeeze(Y[..., 0])
            
        except OSError:
            print("Could not open file " + filename)
            sleep(5)
            
        return X, Y, self.files[idx]


#dataset class for conventional IO
class CamSoftwareDataset(Dataset):

    def __init__(self, filename_filenames, filename_data, filename_label, channels, preprocess = True):
        self.channels = channels

        #preprocessing?
        self.preprocess = preprocess
        
        #read filenames first
        self.files = list(np.load(filename_filenames))

        #shapes
        self.data_shape = (768, 1152, 16)
        self.label_shape = (768, 1152, 2)
        
        #init numpy loader
        #create loaders and parse header
        #data
        self.npr_data = np.memmap(filename_data, dtype='float32', mode='r', offset=128, shape=(len(self.files), 768, 1152, 16))

        #label
        self.npr_label = np.memmap(filename_label, dtype='float32', mode='r', offset=128, shape=(len(self.files), 768, 1152, 2))
        
        #label
        self.length = self.npr_data.shape[0]
        assert(self.length == self.npr_label.shape[0])
        assert(self.length == len(self.files))

        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.data_shape, self.label_shape
    
    def __getitem__(self, idx):

        #get data
        X = np.copy( self.npr_data[idx, ...] )
        if self.preprocess:
            X = np.transpose( X[..., self.channels], (2, 0, 1) )

        #get label
        Y = np.copy( self.npr_label[idx, ...] )
        if self.preprocess:
            Y = np.squeeze( Y[..., 0] )

        return X, Y, self.files[idx]


#dataloader class
class CamDataloader(object):
  
    def __init__(self, filename_filenames, filename_data, filename_label, channels, batchsize, shuffle=False, num_inter_threads = 1, num_intra_threads = 1, device = -1, preprocess = True):
        #channels
        self.channels = channels

        #preprocessing on?
        self.preprocess = preprocess
        
        #read filenames first
        self.files = np.load(filename_filenames)
        
        #init numpy loader
        self.shuffle = shuffle
        self.batchsize = batchsize
        #create loaders and parse header
        #data
        self.npr_data = nr.numpy_reader(split_axis = True, device = device)
        self.npr_data.num_inter_threads = num_inter_threads
        self.npr_data.num_intra_threads = num_intra_threads
        self.npr_data.parse(filename_data)
        self.npr_data.set_batchsize(self.batchsize)
        self.npr_data.init_file(filename_data)

        #label
        self.npr_label = nr.numpy_reader(split_axis = True, device = device)
        self.npr_label.num_inter_threads = num_inter_threads
        self.npr_label.num_intra_threads = num_intra_threads
        self.npr_label.parse(filename_label)
        self.npr_label.set_batchsize(self.batchsize)
        self.npr_label.init_file(filename_label)

        #label
        self.length = self.npr_data.num_samples
        assert(self.length == self.npr_label.num_samples)
        assert(self.length == self.files.shape[0])
        
        #crete permutation array
        self.indices = range(0, self.length)
        if self.shuffle: 
            self.indices = np.random.permutation(self.indices)
        
        print("Initialized data loader with ", self.length, " samples.")

    @property
    def shapes(self):
        return self.npr_data.shape, self.npr_label.shape
        
    def __del__(self):
        self.npr_data.finalize_file()
        self.npr_label.finalize_file()
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for index in range(0, self.length, self.batchsize):
            if (index+self.batchsize) > self.length:
                break
            
            #get batch indices
            batch = self.indices[index : index+self.batchsize]
            
            #grab the data batch and preprocess
            data = self.npr_data.get_batch(batch)

            if self.preprocess:
                if self.batchsize == 1:
                    data = data.unsqueeze(0)
                data = data[..., self.channels].permute(0,3,1,2)
            
            #grab label batch and preprocess
            label = self.npr_label.get_batch(batch)

            if self.preprocess:
                label = torch.squeeze(label[...,0])
                if self.batchsize == 1:
                    label = label.unsqueeze(0)
            
            #grab filenames
            filenames = self.files[batch]
            
            yield data, label, filenames


def split(source, channels, subset_length, train, val, test=0):
    """Get a data split.

    Arguments:
    source -- string containing source directory
    channels -- list containing channels to use
    subset_length -- length of the temporal subset we draw the samples from
    train_size -- fraction of set to use for training
    validation_size -- fraction of set to use for validation
    test_size -- fraction of set to use for testing
    """
    
    dataset = CamDataset(source, channels)
    subset = torch.utils.data.Subset(dataset, range(0, subset_length))
    
    train_size = int(np.floor(train * subset_length))
    val_size = int(np.floor(val * subset_length))
    test_size = int(np.floor(test * subset_length))
    total = train_size + val_size + test_size
    
    train, val, test, _  = torch.utils.data.random_split(subset, [train_size, val_size, test_size, subset_length - total])
    print("Training on ", len(train), " samples, validating on ", len(val), " samples, testing on ", len(test), " samples.")
      
    return train, val, test


def subset(source, channels, start, end, i=1):
    """Get data from start index (inclusively) to end index (exclusively), only using every ith picture.

    Arguments:
    source -- string containing source directory
    channels -- list containing channels to use
    start -- the start index
    end -- the end index
    i -- using every ith picture (default=1)
    """
    
    dataset = CamDataset(source, channels)
    if end < 0:
        end = ( end + len(dataset) ) % len(dataset) + 1
    data = torch.utils.data.Subset(dataset, range(start, end, i))
    
    print("Data subset with ", len(data), " samples.")
      
    return data
