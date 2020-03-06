#these are basic modules
import itertools as it
import numpy as np
import os
import sys
import torch

#test module
import pytest

#this is what we want to test
import numpy_reader as nr



#small helper functions
@pytest.mark.skip(reason="this is a helper function used by the other tests")
def get_sample(reader, filename, sample_id=0):
    reader.parse(filename)
    reader.init_file(filename)
    sample = reader.get_sample(sample_id)
    reader.finalize_file()
    return sample


@pytest.mark.skip(reason="this is a helper function used by the other tests")
def get_diff(array, tensor, devid):
    if devid == -1:
        return np.testing.assert_array_almost_equal(array, tensor.numpy())
    else:
        return np.testing.assert_array_almost_equal(array, tensor.cpu().numpy())


#single sample per file
@pytest.mark.parametrize( "ninter", [1,2,4,8] )
@pytest.mark.parametrize( "nintra", [1,2,4,8] )
def test_single_sample_loads(filename_all, ninter, nintra, devopt):
        
    #copy over
    filename = filename_all
    device_id = devopt
    
    #init reader
    npr = nr.numpy_reader(False, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra
        
    #read files
    array = np.load(filename)
    tensor = get_sample(npr, filename)
        
    #do the assert
    get_diff(array, tensor, device_id)


#multiple samples per file: here we have to distinguish between column major and row major
#row-major
@pytest.mark.parametrize( "ninter", [1,2,4,8] )
@pytest.mark.parametrize( "nintra", [1,2,4,8] )
def test_multi_sample_loads_row_major(filename_row_major, ninter, nintra, devopt):
        
    #copy over
    filename = filename_row_major
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra
        
    #parse file
    npr.parse(filename)
        
    #init file
    npr.init_file(filename)
        
    #read files
    array = np.load(filename)
        
    #get number of samples:
    if not array.shape:
        numsamples = 1
        array = np.array([array])
    else: 
        numsamples = array.shape[0]

    #loop
    for sample in range(numsamples):
        #load tensor
        tensor = npr.get_sample(sample)
            
        #do the assert
        get_diff(array[sample,...], tensor, device_id)
        
    #finalize file
    npr.finalize_file()


#column-major
@pytest.mark.parametrize( "ninter", [1] )
@pytest.mark.parametrize( "nintra", [1] )
def test_multi_sample_loads_column_major(filename_column_major, ninter, nintra, devopt):
        
    #copy over
    filename = filename_column_major
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra
        
    #parse file
    with pytest.raises(RuntimeError) as e:
        npr.parse(filename)
    
    assert "reading column-major arrays" in str(e.value)


#unsupported datatype
@pytest.mark.parametrize( "ninter", [1] )
@pytest.mark.parametrize( "nintra", [1] )
def test_unsupported_datatype(filepathopt, ninter, nintra, devopt):
        
    #copy over
    filename = os.path.join(filepathopt, "arr_1d_dtype_fail.npy")
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra
        
    #parse file
    with pytest.raises(RuntimeError) as e:
        npr.parse(filename)
    
    assert "unsupported datatype" in str(e.value)


#file does not exist
def test_file_does_not_exist(filepathopt, devopt):
        
    #copy over
    filename = os.path.join(filepathopt, "does_not_exist_and_never_did.npy")
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
        
    #parse file
    with pytest.raises(RuntimeError) as e:
        npr.parse(filename)
    
    assert "failed to open file" in str(e.value)


#file exists but header is corrupted
def test_corrupted_header(filepathopt, devopt):
        
    #copy over
    filename = os.path.join(filepathopt, "arr_1d_header_fail.npy")
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
        
    #parse file
    with pytest.raises(RuntimeError) as e:
        npr.parse(filename)
    
    assert "ill formatted or corrupt" in str(e.value)


#file exists but wrong format
def test_wrong_file_format(filepathopt, devopt):
        
    #copy over
    filename = os.path.join(filepathopt, "arr_1d_format_fail.npy")
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(True, device_id)
        
    #parse file
    with pytest.raises(RuntimeError) as e:
        npr.parse(filename)
    
    assert "not a numpy file" in str(e.value)


#file exists, header is OK but payload is corrupted
@pytest.mark.parametrize( "ninter", [1,2,4,8] )
@pytest.mark.parametrize( "nintra", [1,2,4,8] )
def test_corrupted_payload(filepathopt, ninter, nintra, devopt):
        
    #copy over
    filename = os.path.join(filepathopt, "arr_1d_corruption_fail.npy")
    device_id = devopt
        
    #init reader
    npr = nr.numpy_reader(False, device_id)
    npr.num_inter_threads = ninter
    npr.num_intra_threads = nintra
        
    ##read files
    with pytest.raises(IndexError) as e:
        tensor = get_sample(npr, filename)
    
    assert "file corruption" in str(e.value)
