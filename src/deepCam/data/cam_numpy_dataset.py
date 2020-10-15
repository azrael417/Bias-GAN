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
  
    def __init__(self, source, channels, num_intra_threads = 1, device = -1, preprocess = True):
        self.source = source
        self.channels = channels
        self.preprocess = preprocess
        
        self.files = [x.replace("_data.npy", "") for x in sorted(os.listdir(self.source)) if x.endswith("_data.npy")]

        self.length = len(self.files)
        
        #init numpy loader
        filename = os.path.join(self.source, self.files[0])
        #data
        self.npr_data = nr.numpy_reader(split_axis = False, device = device)
        self.npr_data.num_intra_threads = num_intra_threads
        self.npr_data.parse(filename + "_data.npy")
        #label
        self.npr_label = nr.numpy_reader(split_axis = False, device = device)
        self.npr_label.num_intra_threads = num_intra_threads
        self.npr_label.parse(filename + "_label.npy")        
        
        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.npr_data.shape, self.npr_label.shape

    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])
        
        try:
            #load data
            self.npr_data.init_file(filename + "_data.npy")
            X = self.npr_data.get_sample(0)
            self.npr_data.finalize_file()

            #load label
            self.npr_label.init_file(filename + "_label.npy")
            Y = self.npr_label.get_sample(0)
            self.npr_label.finalize_file()
            
        except OSError:
            print("Could not open file " + filename)
            sleep(5)
            
        #preprocess
        if self.preprocess:
            X = X[..., self.channels].permute(2, 0, 1)
            Y = torch.squeeze(Y[..., 0])
            
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


def subset(source, channels, start, end, i=1, num_intra_threads = 1, device = -1, hvd_comm = None, preprocess = True):
    """Get data from start index (inclusively) to end index (exclusively), only using every ith picture.

    Arguments:
    source -- string containing source directory
    channels -- list containing channels to use
    start -- the start index
    end -- the end index
    i -- using every ith picture (default=1)
    """
    
    dataset = CamDataset(source, channels, num_intra_threads, device, preprocess)
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
    
    print("Data subset with ", len(data), " samples.")
      
    return data
