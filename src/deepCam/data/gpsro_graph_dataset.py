import sys
import os
import numpy as np
from time import sleep

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy import sparse

class GraphBatch:

    def __init__(self, tensors, fuse_batch_dim_laplacian = True):
        # extract the numpy arrays
        laplacian = [t[0] for t in tensors]
        data = [t[1] for t in tensors]
        label = [t[2] for t in tensors]
        files = [t[3] for t in tensors]

        # data and label can just be padded:
        np_max = max([t.shape[0] for t in data])
        pad_len = [np_max-t.shape[0] for t in data]
        data = [np.pad(t, ((0, pl), (0,0)), mode='constant', constant_values=0) for pl, t in zip(pad_len, data)]
        label = [np.pad(t, ((0, pl), (0,0)), mode='constant', constant_values=0) for pl, t in zip(pad_len, label)]
        mask = []
        for pl in pad_len:
            tmp = np.ones((np_max), dtype=np.int32)
            tmp[-pl:] = 0
            mask.append(pl)

        # laplacian we need to create a coo tensor
        if not fuse_batch_dim_laplacian:
            # create a (batch_size, np_max, np_max) shapes laplacian
            indices = np.concatenate([np.stack([np.full((t.nnz), idt, dtype=np.int64), t.row.astype(np.int64), t.col.astype(np.int64)], axis=0) for idt, t in enumerate(laplacian)], axis=1)
            values = np.concatenate([t.data for idt, t in enumerate(laplacian)], axis=0)
            lap_shape = (len(tensors), np_max, np_max)
        else:
            # blow laplacian up to (np_max * batch_size, np_max * batch_size), i.e. block diagonal in batch_size:
            indices = np.concatenate([np.stack([t.row.astype(np.int64) + idt * np_max, t.col.astype(np.int64) + idt * np_max], axis=0) for idt, t in enumerate(laplacian)], axis=1)
            values = np.concatenate([t.data for idt, t in enumerate(laplacian)], axis=0)
            lap_shape =	(np_max * len(tensors), np_max * len(tensors))
            
        # stack all the stuff and convert to tensors
        self.data = torch.tensor(np.stack(data, axis=0), dtype = torch.float32)
        self.label = torch.tensor(np.stack(label, axis=0), dtype = torch.float32)
        self.mask = torch.tensor(np.stack(mask, axis=0), dtype = torch.int32)
        self.laplacian = torch.sparse_coo_tensor(indices, values, size=lap_shape, dtype = torch.float32).coalesce()
        self.files = files

    def __call__(self):
        return self.laplacian, self.data, self.label, self.mask, self.files
        
    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        self.mask = self.mask.pin_memory()
        return self


def gpsro_collate_fn(tensors):
    batch = GraphBatch(tensors)
    return batch


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
                 normalization_type = "MinMax",
                 shuffle = True, shard_idx = 0, shard_num = 1,
                 num_intra_threads = 1, seed = 12345):
        
        self.normalization_type = normalization_type
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.shard_idx = shard_idx
        self.shard_num = shard_num

        #set up files
        self.init_files(source)

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
        self.data_shift = np.expand_dims(data_shift.astype(np.float32), axis=0)
        self.data_scale = np.expand_dims(data_scale.astype(np.float32), axis=0)
        self.label_shift = np.expand_dims(label_shift.astype(np.float32), axis=0)
        self.label_scale = np.expand_dims(label_scale.astype(np.float32), axis=0)
        
        print("Initialized dataset with ", self.length, " samples.")

        
    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):
        # open file
        infile = os.path.realpath(os.path.join(self.source, self.files[idx]))
        token = np.load(infile)
        
        # graph
        lap_col = token['laplacian_col']
        lap_row = token['laplacian_row']
        lap_data = token['laplacian_data']
        
        #data
        data = token['data']
        
        #label
        label = token['label']
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)
        label = self.label_scale * (label - self.label_shift)

        # create laplacian csr matrix
        num_points = data.shape[0]
        laplacian = sparse.coo_matrix((lap_data, (lap_row, lap_col)), shape=(num_points, num_points))

        return (laplacian, data, label, self.files[idx])
