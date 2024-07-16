import numpy as np
import h5py
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
import pdb
import os

# # Load the geologic units shapefile
# geologic_units_path = '/home/sujaynair/GMNA_SHAPES/Geologic_units.shp'
# gdf = gpd.read_file(geologic_units_path)

# # Print the current CRS
# print("Current CRS:", gdf.crs)

# # Define the custom CRS
# custom_crs = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=-100 +k=0.926 +x_0=0 +y_0=0 +a=6371204 +b=6371204 +units=m +datum=WGS84")
# gdf = gdf.set_crs(custom_crs, allow_override=True)

# # Print the custom CRS
# print("Custom CRS:", gdf.crs)

# # Transform to EPSG:4326 (latitude and longitude)
# gdf = gdf.to_crs('EPSG:4326')

# # Verify the transformation
# print("Transformed CRS:", gdf.crs)
# print(gdf.geometry.head())

# # Load the countries shapefile
# countries_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
# countries = gpd.read_file(countries_path)

# # Ensure the countries GeoDataFrame is in EPSG:4326
# countries = countries.to_crs('EPSG:4326')

# # Filter to get only the USA
# usa = countries[countries['ADMIN'] == 'United States of America']

# # Crop the geologic units to the USA boundary
# gdf_usa = gpd.overlay(gdf, usa, how='intersection')

# # Define the age mapping dictionary with approximate ages in millions of years ago (Ma)
# age_mapping = {
#     'Paleozoic': 541,
#     'Quaternary': 2.6,
#     'Precambrian': 4500,
#     'Silurian': 443,
#     'Tertiary': 66,
#     'Ordovician': 485,
#     'Late Proterozoic': 1000,
#     'Paleogene': 66,
#     'Early Proterozoic': 2500,
#     'Middle Devonian': 393,
#     'Late Archean': 2800,
#     'Middle Proterozoic': 1600,
#     'Upper Devonian': 382,
#     'Cretaceous': 145,
#     'Lower Cretaceous': 145,
#     'Devonian': 419,
#     'Cambrian': 541,
#     'Upper Cretaceous': 100,
#     'Neogene': 23,
#     'Triassic': 252,
#     'Pliocene': 5.3,
#     'Middle Eocene': 48,
#     'Miocene': 23,
#     'Lower Permian': 299,
#     'Upper Eocene': 33.9,
#     'Oligocene': 33.9,
#     'Paleocene': 66,
#     'Lower Silurian': 443,
#     'Upper Mississippian': 330,
#     'Lower Mississippian': 359,
#     'Lower Eocene': 56,
#     'Upper Pennsylvanian': 299,
#     'Upper Ordovician': 443,
#     'Lower Pennsylvanian': 323,
#     'Jurassic': 201,
#     'Eocene': 56,
#     'Pleistocene': 2.6,
#     'Upper Permian': 252,
#     'Early Cretaceous': 145,
#     'Late Cretaceous': 100,
#     'Mesozoic': 252,
#     'Permian': 299,
#     'Mississippian': 359,
#     'Age unknown': -1,
#     'Mid-Cretaceous': 100,
#     'Pennsylvanian': 323,
#     'Archean': 4000,
#     'Pre-Cretaceous': 145,
#     'Upper Jurassic': 150,
#     'Holocene': 0.01,
#     'Lower Devonian': 419,
#     'Middle Jurassic': 174,
#     'Late Jurassic': 145,
#     'Lower Jurassic': 201,
#     'Middle Cretaceous': 100,
#     'Middle Cambrian': 513,
#     'Maastrichtian': 70,
#     'Proterozoic': 2500,
#     'Middle Archean': 3600,
#     'Early Archean': 3800,
#     'Middle Ordovician': 470,
#     'Pre-Cenozoic undivided': 450,
#     'Lower Ordovician': 485,
#     'Pre-Mesozoic': 541,
#     'Pre-Cenozoic': 66,
#     'Upper Cambrian': 485,
#     'Campanian': 83,
#     'Middle Silurian': 428,
#     'Middle Permian': 265,
#     'Middle Triassic': 237,
#     'Middle Proterozoic': 1600,
#     'Lower Proterozoic': 2500,
#     'Upper Proterozoic': 750,
#     'Pre-Cambrian': 4500,
# }

# # Define the rock type mapping dictionary
# rocktype_mapping = {
#     'Sedimentary': 0,
#     'Plutonic': 1,
#     'Metamorphic and undivided crystalline': 2,
#     'Volcanic': 3,
#     'Ice': 4
# }

# # Function to get geologic unit information for a given lat/lon
# def get_geologic_info(lat, lon):
#     point = Point(lon, lat)
#     result = gdf_usa[gdf_usa.geometry.contains(point)]
#     if not result.empty:
#         min_age_str = result.iloc[0]['MIN_AGE']
#         max_age_str = result.iloc[0]['MAX_AGE']
#         rocktype_str = result.iloc[0]['ROCKTYPE']
#         min_age = age_mapping[min_age_str]
#         max_age = age_mapping[max_age_str]
#         rocktype = rocktype_mapping[rocktype_str]
#         return min_age, max_age, rocktype
#     else:
#         return None, None, None

# # Load the coordinates from the HDF5 file
# h5_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
# with h5py.File(h5_file_path, 'r') as h5f:
#     coords = h5f['coordinates'][:10000]  # Load all 10,000 coordinates

# # Initialize the output array
# output_array = np.zeros((10000, 3, 50, 50))

# # Loop through each grid
# for i, (lat, lon) in enumerate(coords):
#     for x in range(50):
#         for y in range(50):
#             # Calculate the center of the 1-mile by 1-mile cell
#             cell_lat = lat + (x + 0.5) / 69.0  # 1 mile is approximately 1/69 degrees latitude
#             cell_lon = lon + (y + 0.5) / (69.0 * np.cos(np.radians(lat)))  # adjust for longitude based on latitude
#             # Get the geologic info
#             min_age, max_age, rocktype = get_geologic_info(cell_lat, cell_lon)
#             # Fill the output array
#             output_array[i, 0, x, y] = min_age if min_age is not None else 0  # Use 0 to indicate missing data
#             output_array[i, 1, x, y] = max_age if max_age is not None else 0
#             output_array[i, 2, x, y] = rocktype if rocktype is not None else 0
#     if (i + 1) % 100 == 0:
#         print(f"Processed {i + 1} out of 10,000 grids")

# # Save the output array to an HDF5 file
# output_h5_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithGeoInfo.h5'
# with h5py.File(output_h5_file_path, 'w') as h5f:
#     h5f.create_dataset('geoinfo', data=output_array)

# print(f"Geologic information for all 10,000 grids saved to {output_h5_file_path}")
