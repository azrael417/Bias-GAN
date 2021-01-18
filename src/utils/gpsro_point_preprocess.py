import os
import sys
import re
import itertools
import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline as interp
from scipy.sparse import coo_matrix
from tqdm import tqdm
import stripy

# plotting
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# some variables
file_root = "/data/gpsro_data_hires/raw"
level_file = "./gpsro_metadata.csv"
output_dir = "/data/gpsro_data_hires/preproc_point"
pattern = re.compile("data_in_raw_(.*?).data")
tags = {pattern.match(x).groups()[0] for x in os.listdir(file_root) if pattern.match(x) is not None}
earth_radius = 6371000

# special settings
matrix_shape = (91, 181) # 2 degree resolution
degree = 2.
levels = list(range(20,65))
altitude_tag = "geometric_altitude"

# create directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# open level file and select
metadf = pd.read_csv(level_file)
metadf = metadf[ metadf["level"].isin(levels) ].sort_values(by="level", ascending = True).reset_index(drop = True)
altitudes = [ (earth_radius + x) / earth_radius for x in metadf[ altitude_tag ].values.tolist() ]
    
# iterate over tags
for tag in tqdm(tags):
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data in
    data_in = np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data out
    data_out = np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')

    # compute spherical coordinates
    #phi = [ np.pi * (x + 180.) / 180. for x in lon ]
    #theta = [ np.pi * (90. - x) / 180. for x in lat ]
    phi = [np.radians(x) for x in lon]
    theta = [np.radians(x) for x in lat]

    # go altitude by altitude for 2D delaunay
    xcoords = []
    ycoords = []
    zcoords = []
    areas = []

    #print((phi[0][875], theta[0][875], data_in[0][875]), (phi[0][905], theta[0][905], data_in[0][905]))

    # go altitude-wise, because we need that for the 2D delauney
    for ida, a in enumerate(altitudes):

        spherical_triangulation = stripy.sTriangulation(lons=phi[ida], lats=theta[ida])
        all_simplices = spherical_triangulation.simplices

        xcoords.append(spherical_triangulation.x)
        ycoords.append(spherical_triangulation.y)
        zcoords.append(spherical_triangulation.z)
        
        # compute areas:
        area = np.zeros((spherical_triangulation.npoints), dtype=np.float32)
        for pid in range(spherical_triangulation.npoints):

            # get center point
            center_lon, center_lat = spherical_triangulation.lons[pid], spherical_triangulation.lats[pid]
            
            # extract neighboring simplices
            simplices = all_simplices[spherical_triangulation.identify_vertex_triangles([pid])]

            # compute the centroids (i.e. vertices of the dual graph)
            mid_lon, mid_lat = spherical_triangulation.face_midpoints(simplices = simplices)
            
            # iterate through the points and compute the area
            npoints = simplices.shape[0]
            for i in range(npoints):
                triangle_lon = np.array([center_lon, mid_lon[i], mid_lon[(i+1) % npoints]])
                triangle_lat = np.array([center_lat, mid_lat[i], mid_lat[(i+1) % npoints]])
                area[pid] += spherical_triangulation.tri_area(triangle_lon, triangle_lat)

        # append
        areas.append(area)

        ## plot
        #fig = plt.figure(figsize=(20, 10), facecolor="none")
    
        #ax  = plt.subplot(121, projection=ccrs.Mollweide(central_longitude=0.0, globe=None))
        #ax.coastlines(color="#777777")
        #ax.set_global()

        #ax2 = plt.subplot(122, projection=ccrs.Mollweide(central_longitude=0.0,  globe=None))
        #ax2.coastlines(color="#777777")
        #ax2.set_global()

        #lons = np.degrees(spherical_triangulation.lons)
        #lats = np.degrees(spherical_triangulation.lats)
        
        #ax.scatter(lons, lats, color="Red",
        #           marker="o", s=150.0, transform=ccrs.PlateCarree())

        #segs = spherical_triangulation.identify_segments()

        #for s1, s2 in segs:
        #    ax.plot( [lons[s1], lons[s2]],
        #             [lats[s1], lats[s2]], 
        #             linewidth=0.5, color="black", transform=ccrs.Geodetic())

        #plt.savefig(os.path.join(output_dir, "test_mesh.png"))

    # flatten
    xcoords = np.array(list(itertools.chain(*[ x.tolist() for x in xcoords ])), dtype=np.float32)
    ycoords = np.array(list(itertools.chain(*[ y.tolist() for y in ycoords ])), dtype=np.float32)
    zcoords = np.array(list(itertools.chain(*[ z.tolist() for z in zcoords ])), dtype=np.float32)
    areas = np.array(list(itertools.chain(*[ z.tolist() for z in areas ])), dtype=np.float32)
    data_in = np.array(list(itertools.chain(*[ d.tolist() for d in data_in ])), dtype=np.float32)
    data_out = np.array(list(itertools.chain(*[ d.tolist() for d in data_out ])), dtype=np.float32)

    print(xcoords.shape, ycoords.shape, zcoords.shape, areas.shape, data_in.shape, data_out.shape)
    
    # compute final arrays
    #data_in = np.stack([xcoords, ycoords, zcoords, data_in], axis = 1)
    #data_out = np.stack([xcoords, ycoords, zcoords, data_out], axis = 1)

    # delaunay 2D
    #for a in metadf[ altitude_tag ].unique():
    #    print a,
    #    mask = np.all(data_in[:, 3] == a)
    #    data_in_tri = data_in[mask]
    #    print(data_in_tri)
    #triangulation = Delaunay(data_in[:, :3],
    #                         qhull_options = "Qt",
    #                         incremental = False,
    #                         furthest_site = False)

    #print(triangulation.simplices)
    
    ## store results
    #np.save(os.path.join(output_dir, "data_in_" + tag + ".npy"), data_in)
    #np.save(os.path.join(output_dir, "data_out_" + tag + ".npy"), data_out)

    sys.exit(1)
