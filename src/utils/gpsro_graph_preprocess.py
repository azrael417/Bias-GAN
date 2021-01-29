import os
import sys
import re
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

# parallelization
import concurrent.futures as cf

def get_coordinates(file_root, tag, altitudes):
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')

    # compute spherical coordinates
    phi = [np.radians(x) for x in lon]
    theta = [np.radians(x) for x in lat]

    # get level descriptor
    mask = np.concatenate([np.full(t.shape[0], ida) for ida,t in enumerate(theta)], axis=0)
    
    # compute cartesian coordinates
    xcoords = np.concatenate([a * np.cos(p) * np.sin(t) for a,p,t in zip(altitudes, phi, theta)], axis=0)
    ycoords = np.concatenate([a * np.sin(p) * np.sin(t) for a,p,t in zip(altitudes, phi, theta)], axis=0)
    zcoords = np.concatenate([a * np.cos(t) for a,t in zip(altitudes, theta)], axis=0)

    return np.stack([xcoords, ycoords, zcoords], axis=1), mask


def get_data(file_root, tag):
    # input
    data_in = np.concatenate(np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1'), axis=0)
    # label
    data_out = np.concatenate(np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1'), axis=0)
    
    return data_in, data_out


def summarize(file_root, tag, altitudes, num_neighbors):

    # we need this
    from sklearn.neighbors import NearestNeighbors
    
    # get coordinates
    coords, _ = get_coordinates(file_root, tag, altitudes)
    num_points = coords.shape[0]
    # compute knn
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    distance_mean = np.mean(distances)

    # get data
    data_in, data_out = get_data(file_root, tag)
    data_in_mean = np.mean(data_in)
    data_in_sq_mean = np.mean(np.square(data_in))
    data_out_mean = np.mean(data_out)
    data_out_sq_mean = np.mean(np.square(data_out))

    # return results
    return np.array([num_points, distance_mean, data_in_mean, data_in_sq_mean, data_out_mean, data_out_sq_mean], dtype = np.float64)


def preprocess(file_root, tag, output_dir, altitudes, num_neighbors, kernel_width):

    # we need that
    import pygsp as pg
    from pygsp.graphs.nngraphs import nngraph as nng
    
    # vertices
    vertices, mask = get_coordinates(file_root, tag, altitudes)

    # data
    data_in, data_out = get_data(file_root, tag)

    # create graph from these:
    graph = nng.NNGraph(features = vertices,
                        standardize = False,
                        metric = 'euclidean',
                        kind = 'knn',
                        k = num_neighbors,
                        kernel = 'exponential',
                        kernel_width = kernel_width)

    # extract laplacian
    l_ind = graph.L.indices
    l_dat = graph.L.data

    # store graph
    np.savez(os.path.join(output_dir, "data_" + tag + ".npz"),
             data = data_in,
             label = data_out,
             level_ids = mask,
             graph_nn_order = num_neighbors,
             laplacian_indices = l_ind,
             laplacian_data = l_dat)

    
def main():
    # some variables
    file_root = "/data/gpsro_data_hires/raw"
    level_file = "./gpsro_metadata.csv"
    stats_dir = "/data/gpsro_data_hires/spherical"
    output_dir = "/data/gpsro_data_hires/spherical/preproc_graph"
    pattern = re.compile("data_in_raw_(.*?).data")
    tags = sorted([pattern.match(x).groups()[0] for x in os.listdir(file_root) if pattern.match(x) is not None])
    earth_radius = 6371000
    
    # special settings
    levels = list(range(20,65))
    altitude_tag = "geometric_altitude"
    num_neighbors = 20
    max_workers = 8
    recompute_stats = False

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # open level file and select
    metadf = pd.read_csv(level_file)
    metadf = metadf[ metadf["level"].isin(levels) ].sort_values(by="level", ascending = True).reset_index(drop = True)
    altitudes = [ earth_radius + x for x in metadf[ altitude_tag ].values.tolist() ]

    if not os.path.isfile(os.path.join(stats_dir, "stats_graph.npy")) or recompute_stats:
        # determine optimal kernel width for data set: set to average of all computed distances
        results = []
        if max_workers > 1: 
            with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
                futures = [executor.submit(summarize, file_root, tag,
                                           altitudes, num_neighbors) for tag in tags]
                for future in tqdm(cf.as_completed(futures)):
                    results.append(future.result())
        else:
            for tag in tqdm(tags):
                results.append(summarize(file_root, tag, altitudes, num_neighbors))

        # compute statistics:
        results = np.stack(results, axis=0)

        # weight by total points
        weights = np.expand_dims(results[:, 0] / np.sum(results[:, 0]), axis=1)
        stats = np.sum(weights * results[:, 1:], axis=0)

        # store
        np.save(os.path.join(stats_dir, "stats_graph.npy"), stats)
    else:
        stats = np.load(os.path.join(stats_dir, "stats_graph.npy"))

    
    # iterate over tags in parallel fashion
    if max_workers > 1:
        with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
            futures = [executor.submit(preprocess, file_root, tag, output_dir,
                                       altitudes, num_neighbors, stats[0]) for tag in tags]
        
            for future in tqdm(cf.as_completed(futures)):
                continue
    else:
        for tag in tqdm(tags):
            preprocess(file_root, tag, output_dir, altitudes, num_neighbors, stats[0])
            

if __name__ == "__main__":
    main()
