# import os
# import pandas as pd
# import numpy as np
# import geopandas as gpd
# import pickle
# import pdb

# x_min, x_max = -125, -66.5
# y_min, y_max = 24.5, 49.5
# grid_size = 30
# x = np.linspace(x_min, x_max, grid_size + 1)
# y = np.linspace(y_min, y_max, grid_size + 1)
# pixel_size_x = (x_max - x_min) / grid_size
# pixel_size_y = (y_max - y_min) / grid_size

# def create_layers(elem_df, elem):
#     z_A = np.zeros((grid_size, grid_size))
#     z_B = np.zeros((grid_size, grid_size))
#     z_C = np.zeros((grid_size, grid_size))
#     z_D = np.zeros((grid_size, grid_size))
#     z_E = np.zeros((grid_size, grid_size))

#     for i in range(grid_size):
#         for j in range(grid_size):
#             lon_min = x_min + j * pixel_size_x
#             lon_max = lon_min + pixel_size_x
#             lat_max = y_max - i * pixel_size_y
#             lat_min = lat_max - pixel_size_y
#             print(f"Pixel [{i}, {j}] boundaries: Longitude ({lon_min}, {lon_max}), Latitude ({lat_min}, {lat_max})")
#             for _, row in elem_df.iterrows():
#                 if lon_min <= row['longitude'] < lon_max and lat_min <= row['latitude'] < lat_max:
#                     print(f"Yes: ({row['score']}) quality {elem} resource at {row['site_name']}, {row['county']}, {row['state']} at ({row['longitude']}, {row['latitude']}) is in pixel [{i}, {j}]")
#                     if row['score'] == 'A':
#                         z_A[i, j] += 1
#                         z_B[i, j] += 1
#                         z_C[i, j] += 1
#                         z_D[i, j] += 1
#                         z_E[i, j] += 1
#                     elif row['score'] == 'B':
#                         z_B[i, j] += 1
#                         z_C[i, j] += 1
#                         z_D[i, j] += 1
#                         z_E[i, j] += 1
#                     elif row['score'] == 'C':
#                         z_C[i, j] += 1
#                         z_D[i, j] += 1
#                         z_E[i, j] += 1
#                     elif row['score'] == 'D':
#                         z_D[i, j] += 1
#                         z_E[i, j] += 1
#                     elif row['score'] == 'E':
#                         z_E[i, j] += 1

#     return np.stack([z_A, z_B, z_C, z_D, z_E], axis=0)


# file_path = '/Users/sujaynair/Documents/mrds-csv/mrds.csv'
# df = pd.read_csv(file_path)
# df['region'] = df['region'].fillna('NA')
# values = {"commod1": "", "commod2": "", "commod3": ""}
# df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].fillna(value=values)
# df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].astype(str)
# df['commodities'] = df.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
# df = df.assign(commodities=df['commodities'].str.split(',')).explode('commodities')
# df = df[df['commodities'] != '']
# df = df[df['dev_stat'] != 'Plant']
# elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
# data_dir = 'prepared_data'
# os.makedirs(data_dir, exist_ok=True)

# for elem in elements:
#     elem_df = df[df['commodities'].str.strip() == elem]
#     sampled_elem_df = elem_df.sample(frac=1, random_state=1)
#     layers = create_layers(sampled_elem_df, elem)
#     with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'wb') as f:  # MAKE SURE TO CHANGE THIS WHEN CHANGING PERCENT
#         pickle.dump(layers, f)
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
import pickle
import matplotlib.pyplot as plt

# Define bounds and grid size
x_min, x_max = -125, -66.5
y_min, y_max = 24.5, 49.5
grid_size = 60
x = np.linspace(x_min, x_max, grid_size + 1)
y = np.linspace(y_min, y_max, grid_size + 1)
pixel_size_x = (x_max - x_min) / grid_size
pixel_size_y = (y_max - y_min) / grid_size

# Read the shapefile
shapefile_path = '/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
countries = gpd.read_file(shapefile_path)
usa = countries[countries['ADMIN'] == 'United States of America']

# Create the grid cells
grid_cells = []
for i in range(grid_size):
    for j in range(grid_size):
        lon_min = x_min + j * pixel_size_x
        lon_max = lon_min + pixel_size_x
        lat_max = y_max - i * pixel_size_y
        lat_min = lat_max - pixel_size_y
        cell = box(lon_min, lat_min, lon_max, lat_max)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame({'geometry': grid_cells})
grid.crs = usa.crs

# Clip grid cells to the US boundary to identify those fully inside the USA
grid_clipped = gpd.clip(grid, usa)

# Create a mask for cells fully outside the US boundary
mask = np.ones((grid_size, grid_size), dtype=bool)
outside_cells = []

for i in range(grid_size):
    for j in range(grid_size):
        cell = grid.geometry[i * grid_size + j]
        if not any(grid_clipped.intersects(cell)):
            mask[i, j] = False
            lon_min = x_min + j * pixel_size_x
            lon_max = lon_min + pixel_size_x
            lat_max = y_max - i * pixel_size_y
            lat_min = lat_max - pixel_size_y
            outside_cells.append((i, j, lon_min, lon_max, lat_min, lat_max))

# Print cells that are fully outside the USA
print("Cells that are fully outside the USA:")
for cell in outside_cells:
    print(f"Pixel [{cell[0]}, {cell[1]}] boundaries: Longitude ({cell[2]}, {cell[3]}), Latitude ({cell[4]}, {cell[5]})")

# Function to create layers for each element
def create_layers(elem_df, elem):
    z_A = np.zeros((grid_size, grid_size))
    z_B = np.zeros((grid_size, grid_size))
    z_C = np.zeros((grid_size, grid_size))
    z_D = np.zeros((grid_size, grid_size))
    z_E = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            lon_min = x_min + j * pixel_size_x
            lon_max = lon_min + pixel_size_x
            lat_max = y_max - i * pixel_size_y
            lat_min = lat_max - pixel_size_y
            print(f"Pixel [{i}, {j}] boundaries: Longitude ({lon_min}, {lon_max}), Latitude ({lat_min}, {lat_max})")
            for _, row in elem_df.iterrows():
                if lon_min <= row['longitude'] < lon_max and lat_min <= row['latitude'] < lat_max:
                    print(f"Yes: ({row['score']}) quality {elem} resource at {row['site_name']}, {row['county']}, {row['state']} at ({row['longitude']}, {row['latitude']}) is in pixel [{i}, {j}]")
                    if row['score'] == 'A':
                        z_A[i, j] += 1
                        z_B[i, j] += 1
                        z_C[i, j] += 1
                        z_D[i, j] += 1
                        z_E[i, j] += 1
                    elif row['score'] == 'B':
                        z_B[i, j] += 1
                        z_C[i, j] += 1
                        z_D[i, j] += 1
                        z_E[i, j] += 1
                    elif row['score'] == 'C':
                        z_C[i, j] += 1
                        z_D[i, j] += 1
                        z_E[i, j] += 1
                    elif row['score'] == 'D':
                        z_D[i, j] += 1
                        z_E[i, j] += 1
                    elif row['score'] == 'E':
                        z_E[i, j] += 1

    return np.stack([z_A, z_B, z_C, z_D, z_E], axis=0)

# File path and data preparation
file_path = '/Users/sujaynair/Documents/mrds-csv/mrds.csv'
df = pd.read_csv(file_path)
df['region'] = df['region'].fillna('NA')
values = {"commod1": "", "commod2": "", "commod3": ""}
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].fillna(value=values)
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].astype(str)
df['commodities'] = df.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
df = df.assign(commodities=df['commodities'].str.split(',')).explode('commodities')
df = df[df['commodities'] != '']
df = df[df['dev_stat'] != 'Plant']
elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
data_dir = 'prepared_dataClipped60'
os.makedirs(data_dir, exist_ok=True)

for elem in elements:
    elem_df = df[df['commodities'].str.strip() == elem]
    sampled_elem_df = elem_df.sample(frac=1, random_state=1)
    layers = create_layers(sampled_elem_df, elem)
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'wb') as f:
        pickle.dump(layers, f)

print("Data preparation completed.")

# Print cells that are fully outside the USA and their counts
print("Cells that are fully outside the USA and their counts:")
for cell in outside_cells:
    i, j, lon_min, lon_max, lat_min, lat_max = cell
    print(f"Pixel [{i}, {j}] boundaries: Longitude ({lon_min}, {lon_max}), Latitude ({lat_min}, {lat_max})")
    print(f"Counts - A: {layers[0][i, j]}, B: {layers[1][i, j]}, C: {layers[2][i, j]}, D: {layers[3][i, j]}, E: {layers[4][i, j]}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
usa.boundary.plot(ax=ax, linewidth=1)
grid.boundary.plot(ax=ax, color='gray', linestyle='--', linewidth=0.5)
grid_clipped.boundary.plot(ax=ax, color='red', linestyle='--', linewidth=0.5)

# Plot cells fully outside the USA
outside_cells_geom = [box(cell[2], cell[4], cell[3], cell[5]) for cell in outside_cells]
gdf_outside_cells = gpd.GeoDataFrame({'geometry': outside_cells_geom})
gdf_outside_cells.boundary.plot(ax=ax, color='blue', linewidth=1)

plt.title('Grid Cells Outside the USA')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.show()
