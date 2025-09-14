import geopandas as gpd
import numpy as np

filepath = "./soucefile/ping.dxf"

linestrings = gpd.read_file(filepath).geometry[0].coords

arr = np.array(linestrings)

arr_dim_reduction = arr[:, 0:2]
np.savetxt("./soucefile/coords.csv", arr_dim_reduction, delimiter=",", fmt="%.2f")
