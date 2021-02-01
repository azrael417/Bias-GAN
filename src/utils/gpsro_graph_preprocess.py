import os
import sys
import re
import itertools
from scipy import sparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# parallelization
import concurrent.futures as cf

def preprocess_laplacian(laplacian):

    def estimate_lmax(laplacian, tol=5e-3):
        r"""Estimate the largest eigenvalue of an operator."""
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol,
                                   ncv=min(laplacian.shape[0], 10),
                                   return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2*tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        r"""Scale the eigenvalues from [0, lmax] to [-scale, scale]."""
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    # preprocess
    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)

    return laplacian
    

def get_coordinates(file_root, tag):
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')

    # compute spherical coordinates
    phi = [np.radians(x) for x in lon]
    theta = [np.radians(x) for x in lat]
    
    # compute cartesian coordinates
    xcoords = [np.cos(p) * np.sin(t) for p,t in zip(phi, theta)]
    ycoords = [np.sin(p) * np.sin(t) for p,t in zip(phi, theta)]
    zcoords = [np.cos(t) for t in theta]

    # stack now:
    coords = [np.stack([x, y, z], axis=1) for x,y,z in zip(xcoords, ycoords, zcoords)]
    pcoords = [np.stack([p, t], axis=1) for p,t in zip(phi, theta)]

    return coords, pcoords


def get_data(file_root, tag):
    # input
    data_in = np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # label
    data_out = np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    
    return data_in, data_out


def summarize(file_root, tag, num_neighbors):

    # we need this
    from sklearn.neighbors import NearestNeighbors
    
    # get coordinates
    coords, _ = get_coordinates(file_root, tag)
    num_points = sum(c.shape[0] for c in coords)
    
    # compute knn
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(coords[0])
    distances, _ = nbrs.kneighbors(coords[0])
    distance_mean = np.mean(distances)

    # get data
    data_in, data_out = get_data(file_root, tag)
    data_in_mean = np.mean([np.mean(x) for x in data_in])
    data_in_sq_mean = np.mean([np.mean(np.square(x)) for x in data_in])
    data_out_mean = np.mean([np.mean(x) for x in data_out])
    data_out_sq_mean = np.mean([np.mean(np.square(x)) for x in data_out])

    # return results
    return np.array([num_points, distance_mean, data_in_mean, data_in_sq_mean, data_out_mean, data_out_sq_mean], dtype = np.float64)


def preprocess(file_root, tag, output_dir, altitudes, num_neighbors, kernel_width, precondition_laplacian):

    # we need that
    import stripy
    import stripy.spherical as sph
    import pygsp as pg
    from pygsp.graphs.nngraphs import nngraph as nng
    
    # vertices
    all_vertices, all_vertices_polar = get_coordinates(file_root, tag)
    master_vertices = all_vertices[0]
    master_vertices_polar = all_vertices_polar[0]
    master_lons, master_lats = master_vertices_polar[:, 0], master_vertices_polar[:, 1] 

    # data
    data_in, data_out = get_data(file_root, tag)

    # align data:
    error_thrown = False
    for ida, a in enumerate(altitudes[1:], start = 1):
        # extract coordinates and triangulate
        coords = all_vertices[ida]
        pcoords = all_vertices_polar[ida]
        lons, lats = pcoords[:, 0], pcoords[:, 1]

        try:
            spherical_triangulation = stripy.sTriangulation(lons=lons, lats=lats)
        except Exception as err:
            print(f"Error triangulating {tag}: {err}. Applying permutation.")
            try:
                spherical_triangulation = stripy.sTriangulation(lons=lons, lats=lats, permute = True)
            except Exception as err:
                print(f"Error triangulating {tag}: {err}. Skipping sample.")
                error_thrown = True
                break
            

        # interpolate
        data_in[ida], _ = spherical_triangulation.interpolate_linear(master_lons, master_lats, data_in[ida])
        data_out[ida], _ = spherical_triangulation.interpolate_linear(master_lons, master_lats, data_out[ida])

    # if error, exit here
    if error_thrown:
        return
        
    # stack
    data_in = np.stack(data_in, axis=-1)
    data_out = np.stack(data_out, axis=-1)
    
    # create graph from these:
    graph = nng.NNGraph(features = master_vertices,
                        standardize = False,
                        metric = 'euclidean',
                        kind = 'knn',
                        k = num_neighbors,
                        kernel = 'exponential',
                        kernel_width = kernel_width)

    # extract
    lap = graph.L.copy()
    
    # precondition if requested
    if precondition_laplacian:
        lap = preprocess_laplacian(lap)

    # convert to coo matrix:
    lap = sparse.coo_matrix(lap)
        
    # extract laplacian
    l_col = lap.col
    l_row = lap.row
    l_dat = lap.data

    # store graph
    np.savez(os.path.join(output_dir, "data_" + tag + ".npz"),
             data = data_in,
             label = data_out,
             phi = master_lons,
             theta = master_lats,
             r = altitudes,
             graph_nn_order = num_neighbors,
             laplacian_col = l_col,
             laplacian_row = l_row,
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
    recompute_stats = True
    precondition_laplacian = True

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # open level file and select
    metadf = pd.read_csv(level_file)
    metadf = metadf[ metadf["level"].isin(levels) ].sort_values(by="level", ascending = True).reset_index(drop = True)
    altitudes = [ earth_radius + x for x in metadf[ altitude_tag ].values.tolist() ]

    if not os.path.isfile(os.path.join(stats_dir, "stats_graph.npy")) or recompute_stats:
        # determine optimal kernel width for data set: set to average of all computed distances
        print("Computing summary statistics")
        results = []
        if max_workers > 1: 
            with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
                futures = [executor.submit(summarize, file_root, tag, num_neighbors) for tag in tags]
                for future in tqdm(cf.as_completed(futures)):
                    results.append(future.result())
        else:
            for tag in tqdm(tags):
                results.append(summarize(file_root, tag, num_neighbors))

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
    print("Preprocessing data")
    if max_workers > 1:
        with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
            futures = [executor.submit(preprocess, file_root, tag, output_dir,
                                       altitudes, num_neighbors, stats[0], precondition_laplacian) for tag in tags]
        
            for future in tqdm(cf.as_completed(futures)):
                continue
    else:
        for tag in tqdm(tags):
            preprocess(file_root, tag, output_dir, altitudes, num_neighbors, stats[0], precondition_laplacian)
            

if __name__ == "__main__":
    main()
