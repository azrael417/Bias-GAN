import os
from shutil import copyfile
import numpy as np
import argparse as ap
from tqdm import tqdm

def main(pargs):
    
    #check path
    inputpath = pargs.input
    outputpaths = pargs.outputs
    suffix = pargs.suffix

    #check files
    filenames = sorted([x.replace("_data.npy", "") for x in os.listdir(inputpath) if x.endswith("_data.npy")])

    #truncate files so that even number of samples in each:
    num_samples = (len(filenames) // len(outputpaths)) * len(outputpaths)
    filenames = filenames[:num_samples]
    
    #create output directories
    for odir in outputpaths:
        try:
            os.makedirs(odir, exist_ok=True)
        except:
            print("Could not create directory {}".format(odir))
            
    #append the files to data in round robin fashion
    arrays_data = []
    arrays_label = []
    arrays_filenames = []
    for idx, filename in tqdm(enumerate(filenames[0:len(outputpaths)])):
        #data
        arr = np.expand_dims(np.load(os.path.join(inputpath, filename+"_data.npy")), axis=0)
        arrays_data.append(arr)
        #label
        arr = np.expand_dims(np.load(os.path.join(inputpath, filename+"_label.npy")), axis=0)
        arrays_label.append(arr)
        #filenames
        arr = np.array([filename])
        arrays_filenames.append(arr)
        
    for idx, filename in tqdm(enumerate(filenames[len(outputpaths):])):
        #data
        arr = np.expand_dims(np.load(os.path.join(inputpath, filename+"_data.npy")), axis=0)
        arrays_data[idx % len(outputpaths)] = np.concatenate( [ arrays_data[idx % len(outputpaths)], arr ], axis=0)
        #label
        arr = np.expand_dims(np.load(os.path.join(inputpath, filename+"_label.npy")), axis=0)
        arrays_label[idx % len(outputpaths)] = np.concatenate( [ arrays_label[idx % len(outputpaths)], arr ], axis=0)
        #filenames
        arrays_filenames[idx % len(outputpaths)] = np.concatenate( [ arrays_filenames[idx % len(outputpaths) ], np.array([filename]) ], axis=0)

    #write files
    for idx in range(len(outputpaths)):
        outputpath = outputpaths[idx]
        #write data
        datafile = os.path.join(outputpath, "data.npy")
        if not os.path.isfile(datafile) or pargs.overwrite:
            np.save(datafile, arrays_data[idx])
        #write labels
        labelfile = os.path.join(outputpath, "label.npy")
        if not os.path.isfile(labelfile) or pargs.overwrite:
            np.save(labelfile, arrays_label[idx])
        #write filenames
        filenamesfile = os.path.join(outputpath, "filenames.npy")
        if not os.path.isfile(filenamesfile) or pargs.overwrite:
            np.save(filenamesfile, arrays_filenames[idx])


if __name__ == '__main__':
    AP = ap.ArgumentParser()
    AP.add_argument("--input", type=str, help="input directory with files")
    AP.add_argument("--suffix", type=str, default="h5", help="file suffix")
    AP.add_argument("--outputs", type=str, nargs='+', help="output list for the npy files")
    AP.add_argument("--overwrite", action='store_true')
    parsed = AP.parse_args()

    main(parsed)
