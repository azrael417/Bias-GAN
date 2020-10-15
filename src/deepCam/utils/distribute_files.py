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
    filenames = sorted([x for x in os.listdir(inputpath) if x.endswith(suffix)])

    #create output directories
    for odir in outputpaths:
        os.makedirs(odir, exist_ok=True)
    
    #cp the files in round robin fashion
    for idx, filename in tqdm(enumerate(filenames)):
        outputpath = outputpaths[idx % len(outputpaths)]
        if not os.path.isfile(os.path.join(outputpath, filename)):
            copyfile(os.path.join(inputpath, filename), os.path.join(outputpath, filename))


if __name__ == '__main__':
    AP = ap.ArgumentParser()
    AP.add_argument("--input", type=str, help="input directory with files")
    AP.add_argument("--suffix", type=str, default="h5", help="file suffix")
    AP.add_argument("--outputs",type=str, nargs='+', help="output list for the npy files")
    parsed = AP.parse_args()

    main(parsed)
