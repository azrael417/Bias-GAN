import os
import numpy as np
import netCDF4 as nc
import pandas as pd
from tqdm import tqdm

#netcdf file reader helpers
#sizes
def get_dimensions(filename):

    #open netcdf file
    with nc.Dataset(filename, "r") as ncf:
        height = len(ncf.dimensions['latitude'])
        width = len(ncf.dimensions['longitude'])
        nt = len(ncf.dimensions['time'])
        
    return (nt, height, width)


#data
def get_data(filename, variables, array, index, data_format):

    #open netcdf file
    with nc.Dataset(filename, "r") as ncf:
        for idx, variable in enumerate(variables):
            tmparr = (ncf[variable][...]).astype(np.float32)
            if data_format == "nchw":
                array[index:index+24,idx,:,:] = tmparr[...]
            else:
                array[index:index+24,:,:,idx] = tmparr[...]
    return 

        
#helper routine to do the fusing
def fuse_to_numpy(outputpath, resultdf, variables, data_format="nchw", overwrite=False):
    
    #first, check if outputpath exists, otherwise create
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)

    #next, we basically fuse over the days: sorth the df again to ensure correct order
    resultdf.sort_values(by=["dataset","year","month","day"], inplace=True)

    #get the sizes to preallocate the array
    dims = get_dimensions(resultdf.loc[0, "filename"])

    #iterate over the years and months but pack all days into a single file
    features = resultdf[["dataset","year","month"]].apply(lambda x: (x["dataset"], x["year"], x["month"]), axis=1).unique()
    for feature in features:
        
        #outputfiles
        outputname = os.path.join(outputpath, "data-"+feature[0]+"-"+str(feature[1])+"-"+str(feature[2])+".npy")
        foutputname = os.path.join(outputpath, "filenames-"+feature[0]+"-"+str(feature[1])+"-"+str(feature[2])+".npy")

        #check if file exists
        if os.path.isfile(outputname) and not overwrite:
            print("file {} already exists".format(outputname))
            continue

        #project df
        selectdf = resultdf[ (resultdf["dataset"] == feature[0]) & (resultdf["year"] == feature[1]) & (resultdf["month"] == feature[2]) ]
        
        #the full dims are: dims[0]*num_files, dims[1], dims[2]
        if data_format == "nchw":
            full_dims = (dims[0]*selectdf.shape[0], len(variables), dims[1], dims[2])
        else:
            full_dims = (dims[0]*selectdf.shape[0], dims[1], dims[2], len(variables))

        #allocate array
        arr = np.zeros( full_dims, dtype=np.float32 )
        flist = []

        #fill it:
        print("preparing {}".format(outputname))
        for idx, day in enumerate(tqdm(selectdf["day"].unique())):
            filename = selectdf.loc[ selectdf["day"] == day, "filename" ].values[0]
            flist += [filename]*24
            get_data(filename, variables, arr, idx*24, data_format)

        #store the stuff
        np.save(outputname, arr)
        np.save(foutputname, np.array(flist))

        
#global parameters
nraid = 4
variables = ["u10", "v10", "d2m", "t2m", "msl", "mwd", "mwp", "sst", "swh", "sp", "tp"]
overwrite = True
data_format = "nchw"

for idx in range(0,nraid):

    #root path
    root = '/data{}/ecmwf_data'.format(2 * idx + 1)
    
    for gpudir in os.listdir(root):

        if (gpudir not in ["gpu0", "gpu1", "gpu2", "gpu3"] ):
            continue
        
        #get all the files, sort
        resultdf = pd.DataFrame([{"dataset":y[0],
                                  "year":y[1].split("-")[0],
                                  "month":y[1].split("-")[1],
                                  "day": y[1].split("-")[2],
                                  "filename": os.path.join(root,gpudir,"_".join(y)+'.nc')}
                                  for y in [ os.path.splitext(x)[0].split("_") for x in os.listdir(os.path.join(root,gpudir)) if x.endswith('.nc') ] ])

        #check if there was any data at all
        if resultdf.empty:
            continue

        #do some transformations
        resultdf[["year","month","day"]] = resultdf[["year","month","day"]].astype(int)
        resultdf.sort_values(by=["dataset","year","month","day"], inplace=True)
        
        #see if there are gaps in the months
        dropdf = resultdf.groupby(["dataset", "year", "month"]).apply(lambda x: x["day"].max() != x["day"].count()).reset_index().rename(columns={0: "drop"})
        resultdf = resultdf.merge(dropdf, on=["dataset", "year", "month"], how="left")

        #remove all non-consecutive months
        resultdf = resultdf[ resultdf["drop"] == False ]
        del resultdf["drop"]

        #do the fusing
        outputpath = os.path.join(root, gpudir, "all")
        fuse_to_numpy(outputpath, resultdf, variables, data_format, overwrite)
        
    break

        
