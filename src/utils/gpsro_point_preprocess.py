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

def preprocess(tag, altitudes, triangulation_type, coord_system):
    
    # load lon
    lon = np.load(os.path.join(file_root,"lon_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load lat
    lat = np.load(os.path.join(file_root,"lat_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data in
    data_in = np.load(os.path.join(file_root,"data_in_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')
    # load data out
    data_out = np.load(os.path.join(file_root,"data_out_raw_"+tag+".data"), allow_pickle=True, encoding='latin1')

    # compute spherical coordinates
    phi = [np.radians(x) for x in lon]
    theta = [np.radians(x) for x in lat]

    # go altitude by altitude for 2D delaunay
    # cartesian
    xcoords = []
    ycoords = []
    zcoords = []
    # spherical
    rcoords = []
    phicoords = []
    thetacoords = []
    # integration measure
    areas = []

    # go altitude-wise, because we need that for the 2D delauney
    for ida, a in enumerate(altitudes):

        # triangulate
        spherical_triangulation = stripy.sTriangulation(lons=phi[ida], lats=theta[ida])
        all_simplices = spherical_triangulation.simplices

        # cartesian coordinates
        xcoords.append(spherical_triangulation.x)
        ycoords.append(spherical_triangulation.y)
        zcoords.append(spherical_triangulation.z)

        # spherical coordinates
        rcoords.append(np.full(spherical_triangulation.lons.shape, a, dtype=np.float32))
        phicoords.append(spherical_triangulation.lons)
        thetacoords.append(spherical_triangulation.lats)

        if triangulation_type == "nodal":
            # get midpoints
            mid_lon, mid_lat = spherical_triangulation.face_midpoints()
            
            # interp
            data_in_interp, _ = spherical_triangulation.interpolate_linear(mid_lon, mid_lat, data_in[ida])
            data_out_interp, _ = spherical_triangulation.interpolate_linear(mid_lon, mid_lat, data_out[ida])
            area = spherical_triangulation.areas()

            # update data array
            data_in[ida] = data_in_interp
            data_out[ida] = data_out_interp
            
        elif triangulation_type == "dual":
            # compute areas:
            area = np.zeros((spherical_triangulation.npoints), dtype=np.float32)
            for pid in range(spherical_triangulation.npoints):
        
                # get center point
                center_lon, center_lat = spherical_triangulation.lons[pid], spherical_triangulation.lats[pid]
         
                # extract neighboring simplices
                simplices = all_simplices[spherical_triangulation.identify_vertex_triangles([pid])]
        
                # compute the centroids (i.e. vertices of the dual graph)
                mid_lon, mid_lat = spherical_triangulation.face_midpoints(simplices = simplices)

                # create new triangulation
                tri = stripy.sTriangulation(lons=np.insert(mid_lon, 0, center_lon), lats=np.insert(mid_lat, 0, center_lat))
                area[pid] = np.sum(tri.areas())

            # normalize:
            area *= (4.*np.pi) / np.sum(area)

        else:
            raise NotImplementedError(f"Error, triangulation type {triangulation_type} is not supported.")

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
    # cartesian
    xcoords = np.array(list(itertools.chain(*[ x.tolist() for x in xcoords ])), dtype=np.float32)
    ycoords = np.array(list(itertools.chain(*[ y.tolist() for y in ycoords ])), dtype=np.float32)
    zcoords = np.array(list(itertools.chain(*[ z.tolist() for z in zcoords ])), dtype=np.float32)

    # spherical
    rcoords = np.array(list(itertools.chain(*[ x.tolist() for x in rcoords ])), dtype=np.float32)
    phicoords = np.array(list(itertools.chain(*[ x.tolist() for x in phicoords ])), dtype=np.float32)
    thetacoords = np.array(list(itertools.chain(*[ x.tolist() for x in thetacoords ])), dtype=np.float32)
    
    # integration measures
    areas = np.array(list(itertools.chain(*[ z.tolist() for z in areas ])), dtype=np.float32)

    # data
    data_in = np.array(list(itertools.chain(*[ d.tolist() for d in data_in ])), dtype=np.float32)
    data_out = np.array(list(itertools.chain(*[ d.tolist() for d in data_out ])), dtype=np.float32)

    # compute final arrays
    if coord_system == "cartesian":
        data_in = np.stack([xcoords, ycoords, zcoords, areas, data_in], axis = 1)
        data_out = np.stack([xcoords, ycoords, zcoords, areas, data_out], axis = 1)
    elif coord_system == "spherical":
        data_in = np.stack([rcoords, phicoords, thetacoords, areas, data_in], axis = 1)
        data_out = np.stack([rcoords, phicoords, thetacoords, areas, data_out], axis = 1)
    else:
        raise NotImplementedError(f"Error, {coord_system} coordinate system not supported!")
    
    # store results
    np.save(os.path.join(output_dir, "data_in_" + tag + ".npy"), data_in)
    np.save(os.path.join(output_dir, "data_out_" + tag + ".npy"), data_out)

def main():
    # some variables
    file_root = "/data/gpsro_data_hires/raw"
    level_file = "./gpsro_metadata.csv"
    output_dir = "/data/gpsro_data_hires/preproc_point"
    pattern = re.compile("data_in_raw_(.*?).data")
    tags = sorted([pattern.match(x).groups()[0] for x in os.listdir(file_root) if pattern.match(x) is not None])
    earth_radius = 6371000

    # special settings
    matrix_shape = (91, 181) # 2 degree resolution
    degree = 2.
    levels = list(range(20,65))
    altitude_tag = "geometric_altitude"
    triangulation_type = "dual"
    coord_system = "spherical"

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # open level file and select
    metadf = pd.read_csv(level_file)
    metadf = metadf[ metadf["level"].isin(levels) ].sort_values(by="level", ascending = True).reset_index(drop = True)
    altitudes = [ (earth_radius + x) / earth_radius for x in metadf[ altitude_tag ].values.tolist() ]
    
    # iterate over tags
    for idt,tag in enumerate(tqdm(tags)):
        preprocess(tag, altitudes, triangulation_type, coord_system)

if __name__ == "__main__":
    main()
