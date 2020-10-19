#these are basic modules
import itertools as it
import numpy as np
import os
import sys
import torch
import horovod.torch as hvd

#this is what we want to test
import numpy_reader as nr


def get_diff(array, tensor, devid):
    if devid == -1:
        return np.testing.assert_array_almost_equal(array, tensor.numpy())
    else:
        return np.testing.assert_array_almost_equal(array, tensor.cpu().numpy())


def get_sample(reader, filename, sample_id=0):
    reader.parse(filename)
    reader.init_file(filename)
    sample = reader.get_sample(sample_id)
    reader.finalize_file()
    return sample

def test_single_sample_loads(filename_all, ninter, nintra, devopt):

    #copy over
    filename = filename_all
    device_id = 0

    #init reader
    npr = nr.numpy_reader(False, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra

    #read files
    array = np.load(filename)
    tensor = get_sample(npr, filename)

    #do the assert
    get_diff(array, tensor, device_id)

    #print results:
    print(tensor.cpu().numpy())
    print(array)


def main():
    hvd.init()
    test_single_sample_loads("/data1/test/arr_1d_fp64_rm.npy", 1, 1, 3)

if __name__ == "__main__":
    main()
