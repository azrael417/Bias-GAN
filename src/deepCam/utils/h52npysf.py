import os
import numpy as np
import h5py as h5
import argparse as ap
from tqdm import tqdm


def main(pargs):
    
    #check path
    inputpath = pargs.input
    outputpath = pargs.output

    #check inputs
    filenames = sorted([x for x in os.listdir(inputpath) if x.endswith(".h5")])

    #create outputpath if doesn't exist:
    os.makedirs(outputpath, exist_ok=True)

    arrays_data = []
    arrays_label = []
    for filename in tqdm(filenames):
        #read input
        inputfile = os.path.join(inputpath, filename)
        with h5.File(inputfile, 'r') as f:
            data = f["climate"]["data"][...].astype(np.float32)
            label = np.stack([f["climate"]["labels_0"][...], f["climate"]["labels_1"][...]], axis=-1)

        #append arrays
        arrays_data.append(data)
        arrays_label.append(label)

    #write output
    #data
    np.save(os.path.join(outputpath, "data.npy"), np.stack(arrays_data, axis=0))

    #label
    np.save(os.path.join(outputpath, "label.npy"), np.stack(arrays_label, axis=0))

    #filenames
    np.save(os.path.join(outputpath, "filenames.npy"), np.stack(filenames, axis=0))


if __name__ == '__main__':
    AP = ap.ArgumentParser()
    AP.add_argument("--input", type=str, help="input directory with hdf5 files")
    AP.add_argument("--output",type=str, help="output directory for the npy files")
    parsed = AP.parse_args()
    
    main(parsed)
