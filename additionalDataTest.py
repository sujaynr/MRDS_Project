import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data from the HDF5 file
input_h5_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithGeoInfo_First10.h5'
with h5py.File(input_h5_file_path, 'r') as h5f:
    geoinfo = h5f['geoinfo'][:]

# Create the directory if it does not exist
output_dir = 'LayerPlots/geoage'
os.makedirs(output_dir, exist_ok=True)

# Define a function to plot the data and save it to a file
def plot_and_save_layer(data, layer_name, grid_num):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label=f'{layer_name}')
    plt.title(f'{layer_name} for Grid {grid_num}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(output_dir, f'{layer_name}_Grid_{grid_num}.png'))
    plt.close()

# Plot the data for each grid and each layer
for grid_num in range(10):
    min_age_data = geoinfo[grid_num, 0, :, :]
    max_age_data = geoinfo[grid_num, 1, :, :]
    rocktype_data = geoinfo[grid_num, 2, :, :]
    
    plot_and_save_layer(min_age_data, 'Min_Age_Ma', grid_num)
    plot_and_save_layer(max_age_data, 'Max_Age_Ma', grid_num)
    plot_and_save_layer(rocktype_data, 'Rock_Type', grid_num)

print(f"Plots saved in {output_dir}")
