import os
import h5py as h5
import numpy as np
from time import sleep

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class CamDataset(Dataset):
  
    def __init__(self, source, channels):
        self.source = source
        self.channels = channels
        self.files = sorted(os.listdir(self.source))
        self.files.remove("ah_data-1996-01-01-00-1.h5")
        self.length = len(self.files)
        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                fin = h5.File(self.source+self.files[idx], "r")
                break
            except OSError:
                print("Could not open file "+self.files[idx])
                sleep(5)
                
        X = fin['climate']['data'][self.channels,...][()]
        y = fin['climate']['labels'][0][()]
        fin.close()
        return X, y
  
def split(source, channels, train, val, test):
    """Get a data split.

    Arguments:
    source -- string containing source directory
    channels -- list containing channels to use
    train -- fraction of set to use for training
    val -- fraction of set to use for validation
    test -- fraction of set to use for testing
    """
    
    dataset = CamDataset(source, channels)
    
    train_size = int(np.floor(train * len(dataset)))
    val_size = int(np.floor(val * len(dataset)))
    test_size = len(dataset) - train_size - val_size
    
    train, val, test  = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    print('Train: ', len(train), ', Val: ', len(val), ', Test: ', len(test))
      
    return train, val, test