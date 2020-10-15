import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#custom reader
import numpy_reader as nr

#dataset class
class ERADataLoader(DataLoader):

    def init_reader(self, filename):
        #get data
        data_file = os.path.realpath(os.path.join(self.directory, "data-"+filename))
        self.npr.parse(data_file)
        self.npr.init_file(data_file)
        #important, since we read label and data from the same file, we need to multiply the effective batch size by 2
        self.npr.set_batchsize(self.batch_size)
        
        #get filenames
        filename_file = os.path.realpath(os.path.join(self.directory, "filenames-"+filename))
        current_filenames = np.load(filename_file)
        
        #get the samples and shuffle if requested
        current_samples = np.array(range(0, self.npr.num_samples))
        if self.shuffle:
            #reshape so it will only permute the leading axis
            current_samples = np.reshape(current_samples, (current_samples.shape[0] // 2, 2))
            current_permutation = self.rng.permutation(current_samples).flatten()
            current_samples = current_samples.flatten()
            current_samples = current_samples[current_permutation]
            current_filenames = current_filenames[current_permutation]
            
        return current_samples, current_filenames

            
    def __init__(self, directory, statsfile, channels, batch_size, \
                 normalization_type = "MinMax", shuffle = True, \
                 read_device = torch.device("cpu"), send_device = torch.device("cpu"), \
                 num_inter_threads = 1, num_intra_threads = 1, seed = 12345):
        #copy some parameters
        self.directory = directory
        self.channels = channels
        self.batch_size = batch_size
        self.normalization_type = normalization_type
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.read_device = read_device
        self.send_device = send_device

        #set up reader
        if (self.read_device == torch.device("cpu")):
            devindex = -1
        else:
            devindex = self.read_device.index
        self.npr = nr.numpy_reader(split_axis = True, device = devindex)
        self.npr.num_inter_threads = num_inter_threads
        self.npr.num_intra_threads = num_intra_threads

        #open statsfile
        statsfile = np.load(statsfile)

        #compute normalization
        if self.normalization_type == "MinMax":
            #get min and max:
            data_shift = statsfile["minval"][self.channels]
            data_scale = 1. / ( statsfile["maxval"][self.channels] - data_shift )
        elif self.normalization_type == "MeanVariance":
            #get <x> and <x**2>:
            data_shift = statsfile["mean"][self.channels]
            data_scale = 1. / np.sqrt( statsfile["sqmean"][self.channels] - np.square(data_shift) )

        #reshape into broadcastable shape
        data_shift = np.reshape( data_shift, (1, data_shift.shape[0], 1, 1) ).astype(np.float32)
        data_scale = np.reshape( data_scale, (1, data_scale.shape[0], 1, 1) ).astype(np.float32)
        #store into tensors
        self.data_shift = torch.tensor(data_shift, requires_grad=False).to(self.send_device)
        self.data_scale	= torch.tensor(data_scale, requires_grad=False).to(self.send_device)
        
        #get list with files
        self.files = [ x.replace("data-","") for x in os.listdir(self.directory) if x.endswith(".npy") and x.startswith("data-") ]

        #determine total length of files
        self.length = 0
        for f in self.files:
            self.npr.parse(os.path.realpath(os.path.join(self.directory, "data-"+f)))
            #divide by 2 bc we have labels and data in the same sample
            self.length += self.npr.num_samples // 2

        #shuffle the files for the first time and init the first guy
        self.rng.shuffle(self.files)
        
        #print status message
        print("Initialized dataset with ", self.length, " samples.")

        
    def __del__(self):
        self.npr.finalize_file()

        
    def __len__(self):
        return self.length

    
    def __iter__(self):
        
        #iterate over files
        for filename in self.files:

            #init reader and get the samples from the file
            samples, filenames = self.init_reader(filename)

            #iterate over samples in the file
            for index in range(0, len(samples), self.batch_size*2):
                if (index+self.batch_size*2) > len(samples):
                    break

                #get batch information
                batch_samples = samples[index:index+self.batch_size*2]
                batch_filenames = filenames[index:index+self.batch_size*2]
                batch_info = list(zip(batch_filenames, batch_samples))
                #split list for easier processing
                data_info = batch_info[0::2]
                label_info = batch_info[1::2]

                #get the data
                data = self.npr.get_batch(batch_samples[0::2])
                label = self.npr.get_batch(batch_samples[1::2])

                #unsqueeze tensor
                if self.batch_size == 1:
                    data = data.unsqueeze(0)
                    label = label.unsqueeze(0)

                #project channels
                data = data[:, self.channels, ...]
                label = label[:, self.channels, ...]

                # upload to other device if necessary
                if data.device != self.send_device:
                    data = data.to(self.send_device)
                if label.device != self.send_device:
                    label = label.to(self.send_device)
                    
                #preprocess
                data = self.data_scale * (data - self.data_shift)
                label = self.data_scale * (label - self.data_shift)
                
                yield data, label, data_info, label_info
