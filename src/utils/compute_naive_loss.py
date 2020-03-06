import os
import numpy as np
import argparse as ap

from mpi4py import MPI


def main(pargs):
    
    #global parameters
    nraid = pargs.num_raid
    data_format = "nchw"
    #data_path_prefix = "/"
    data_dir_prefix = pargs.data_dir_prefix
    channels = pargs.channels
    normalization_file = pargs.normalization_file
    normalization_type = pargs.normalization_type

    #MPI stuff
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    #get list of files on rank 0:
    files= []
    if comm_rank == 0:
        
        #raid drives
        for idx in range(0,nraid):

            #root path
            root = os.path.join( data_dir_prefix, 'data{}'.format(2 * idx + 1), 'ecmwf_data' )

            for gpudir in os.listdir(root):

                #exclude some crappy dirs
                if not gpudir.startswith("gpu"):
                    continue

                files += [ os.path.join(root, gpudir, 'validation', x)  for x in os.listdir(os.path.join(root, gpudir, 'validation')) \
                           if x.endswith('.npy') and x.startswith('data-') ]

        #sort the files
        files.sort()
        
    #broadcast list of files to all ranks
    files = comm.bcast(files, root = 0)

    #compute bounds
    chunksize = int(np.ceil(len(files) / comm_size))
    start_idx = chunksize * comm_rank
    end_idx = min([start_idx + chunksize, len(files)])
    files = files[start_idx:end_idx]

    #load normalization on rank0
    if comm_rank == 0:
        statsfile = np.load(normalization_file)
        if normalization_type == "MinMax":
            #get min and max:
            data_shift = statsfile["minval"][channels]
            data_scale = 1. / ( statsfile["maxval"][channels] - data_shift )
        elif normalization_type == "MeanVariance":
            #get <x> and <x**2>:
            data_shift = statsfile["mean"][channels]
            data_scale = 1. / np.sqrt( statsfile["sqmean"][channels] - np.square(data_shift) )
    else:
        data_shift = np.zeros(len(channels))
        data_scale = np.zeros(len(channels))

    #broadcast
    data_shift = comm.bcast(data_shift, root = 0)
    data_scale = comm.bcast(data_scale, root = 0)
    
    #reshape
    data_shift = data_shift.reshape(1, data_shift.shape[0], 1, 1)
    data_scale = data_scale.reshape(1, data_scale.shape[0], 1, 1)
    
    #start loop
    loss = 0.
    count = 0
    for filename in files:

        #load data and project
        token = np.load(filename)
        token = token[:, channels, ...]
        #increase counter
        current_count = token.shape[0]
        count += current_count
        #normalize
        token = data_scale * (token - data_shift)
        #split in data and label
        data = token[0::2]
        label = token[1::2]
        #compute diff
        absdiff = np.abs(data-label)
        current_loss = np.sum(np.mean(absdiff, axis=(1,2,3)))
        loss += current_loss

    print(comm_rank, count, loss)

    #reduce the loss and count
    loss = comm.reduce(loss, root = 0)
    count = comm.reduce(count, root = 0)
    
    #print result
    if comm_rank == 0:
        print("Final average loss: {} ({} samples)".format(loss / float(count), count))

    
if __name__ == '__main__':

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--num_raid", type=int, default=4, choices=[4, 8], help="Number of available raid drives")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--normalization_file", type=str, default='/', help="NPZ file which contains normalization data")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10], help="Channels used in input")
    AP.add_argument("--normalization_type", type=str, default="MinMax", choices=["MinMax", "MeanVariance"], help="Normalization type")
    pargs = AP.parse_args()
    
    main(pargs)
