import os
import h5py as h5
import numpy as np
from time import sleep

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#dataset class
class CamDataset(Dataset):
  
    def __init__(self, source, channels, preprocess = True):
        self.source = source
        self.channels = channels
        self.preprocess = preprocess
        self.files = sorted(os.listdir(self.source))

        self.length = len(self.files)

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape
        
        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        while(True):
            try:
                fin = h5.File(filename, "r")
                break
            except OSError:
                print("Could not open file " + filename + ", trying again in 5 seconds.")
                sleep(5)

        X = fin['climate']['data'][()]
        if self.preprocess:
            X = X[...,self.channels]
            X = np.moveaxis(X, -1, 0)
        Y = fin['climate']['labels_0'][()]
        fin.close()
        
        return X, Y, filename


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


def subset(source, channels, start, end, i=1, hvd_comm = None, preprocess = True):
    """Get data from start index (inclusively) to end index (exclusively), only using every ith picture.

    Arguments:
    source -- string containing source directory
    channels -- list containing channels to use
    start -- the start index
    end -- the end index
    i -- using every ith picture (default=1)
    """
    
    dataset = CamDataset(source, channels, preprocess)
    if end < 0:
        end = ( end + len(dataset) ) % len(dataset) + 1

    #cut range first and do striding if requested
    data = torch.utils.data.Subset(dataset, range(start, end, i))

    #make sure all the ranks have the same number of samples
    #this voids running in load balancing issues
    if hvd_comm:
        sample_tens = torch.tensor(np.array([len(data)])).detach().cpu()
        sample_tens = hvd_comm.allgather(sample_tens)
        min_samples = np.min(sample_tens.numpy())
        data = torch.utils.data.Subset(data, range(0, min_samples))
    
    print("Data subset with {} samples".format(len(data)))
      
    return data
