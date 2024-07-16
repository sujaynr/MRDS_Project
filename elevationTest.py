import pdb
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import geopandas as gpd

# def read_hgt_file(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             data = np.fromfile(f, dtype='>i2')  # HGT files are big-endian 16-bit integers
#         if data.size == 1201 * 1201:
#             data = data.reshape((1201, 1201))  # HGT files with a resolution of 1201x1201
#         elif data.size == 3601 * 3601:
#             data = data.reshape((3601, 3601))  # HGT files with a resolution of 3601x3601
#         else:
#             raise ValueError(f"Unexpected file size: {data.size}")
#         return data
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return None

# def get_lat_lon_from_filename(filename):
#     lat = int(filename[1:3])
#     lon = int(filename[4:7])
#     if filename[0] == 'S':
#         lat = -lat
#     if filename[3] == 'W':
#         lon = -lon
#     return lat, lon

# def get_elevation(lat, lon, data, lat_start, lon_start):
#     lat_idx = int((lat - lat_start) * (data.shape[0] - 1))
#     lon_idx = int((lon - lon_start) * (data.shape[1] - 1))
#     return data[lat_idx, lon_idx]

# def get_cell_elevation(lat, lon, data, lat_start, lon_start):
#     elevation_nw = get_elevation(lat, lon, data, lat_start, lon_start)
#     elevation_ne = get_elevation(lat, lon + 1 / 1200, data, lat_start, lon_start)
#     elevation_sw = get_elevation(lat + 1 / 1200, lon, data, lat_start, lon_start)
#     elevation_se = get_elevation(lat + 1 / 1200, lon + 1 / 1200, data, lat_start, lon_start)
#     return (elevation_nw + elevation_ne + elevation_sw + elevation_se) / 4

# def process_grid(lat_start, lon_start, directory):
#     result = np.zeros((50, 50))
#     files = [f for f in os.listdir(directory) if f.endswith('.hgt')]
#     for file in files:
#         file_path = os.path.join(directory, file)
#         data = read_hgt_file(file_path)
#         if data is not None:
#             lat, lon = get_lat_lon_from_filename(file)
#             if lat_start <= lat < lat_start + 1 and lon_start <= lon < lon_start + 1:
#                 for i in range(50):
#                     for j in range(50):
#                         cell_lat = lat_start + i / 1200
#                         cell_lon = lon_start + j / 1200
#                         result[i, j] = get_cell_elevation(cell_lat, cell_lon, data, lat, lon)
#                 break
#     return result

# def process_all_grids(coords, directory):
#     num_points = coords.shape[0]
#     all_elevations = np.zeros((num_points, 50, 50))
#     for idx, (lat_start, lon_start) in enumerate(coords):
#         print(f"Processing {idx + 1}/{num_points}: lat={lat_start}, lon={lon_start}")
#         all_elevations[idx] = process_grid(lat_start, lon_start, directory)
#     return all_elevations

# # Load coordinates
# h5_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
# with h5py.File(h5_file_path, 'r') as f:
#     coords = f['coordinates'][:]

# # Process grids
# directory = '/home/sujaynair/MRDS_Project/elevation_data'
# all_elevations = process_all_grids(coords, directory)

# # Save results
# output_path = 'all_elevations.h5'
# with h5py.File(output_path, 'w') as f:
#     f.create_dataset('elevations', data=all_elevations)

# print(f"All elevations saved at {output_path}")

import numpy as np
import matplotlib.pyplot as plt

file = '/home/sujaynair/MRDS_Project/all_elevations.h5'
with h5py.File(file, 'r') as f:
    elevations = f['elevations'][:]
coords = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
with h5py.File(coords, 'r') as f:
    coords = f['coordinates'][:]

countries_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
countries = gpd.read_file(countries_path)
us = countries[countries['ADMIN'] == 'United States of America']
miles_to_degrees = 1 / 69.0  # Approximation (1 degree latitude is roughly 69 miles)


# Plot the US map
fig, ax = plt.subplots(figsize=(15, 15))
us.boundary.plot(ax=ax, linewidth=1)
# Overlay the first 20 grids
for idx in range(10000):
    lat, lon = coords[idx]
    elevation_grid = elevations[idx]
    
    # Plot each grid
    extent = [lon, lon + 50 * miles_to_degrees, lat, lat + 50 * miles_to_degrees]
    im = ax.imshow(elevation_grid, cmap='terrain', extent=extent, alpha=0.5)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
cbar.set_label('Elevation (meters)')

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Elevation Maps for the First 20 50x50 Grids Overlaying the US')
ax.set_xlim([-130, -60])  # Adjust as necessary for your data
ax.set_ylim([20, 50])     # Adjust as necessary for your data


plt.savefig('elevation_maps.png')