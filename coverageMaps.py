import h5py
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import random
import argparse
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Argument parser to include heatmap flag
parser = argparse.ArgumentParser(description="Generate coverage maps for minerals and RaCA sites.")
parser.add_argument('--heatmap', action='store_true', help='Generate heatmap instead of scatter plot.')
args = parser.parse_args()

# File paths
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
raca_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagrids.h5'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'

# Load data from HDF5 files
with h5py.File(h5_file_path, 'r') as f:
    coords = f['coordinates'][:]
    counts = f['counts'][:]

with h5py.File(raca_file_path, 'r') as f:
    raca_data = f['racagrids'][:]

# Elements to visualize
elements = ['Gold', 'Silver', 'Nickel']
element_indices = [0, 1, 5]  # Indices corresponding to Gold, Silver, Nickel in 'counts'

# USA shapefile
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']

# Define the grid for the heatmap (Reduced grid size for larger squares)
grid_size = 50  # Reduce the grid size to make squares larger

lon_min, lat_min, lon_max, lat_max = us_shape.total_bounds
lon_bins = np.linspace(lon_min, lon_max, grid_size)
lat_bins = np.linspace(lat_min, lat_max, grid_size)

# Function to create coverage maps
def create_coverage_map(element_index, element_name, coords, counts, raca_data, us_shape, output_path, heatmap=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')

    labels = []

    if heatmap:
        # Initialize heatmap arrays
        heatmap_mineral = np.zeros((grid_size, grid_size))
        heatmap_raca = np.zeros((grid_size, grid_size))

        # Get random indices for minerals
        mineral_indices = random.sample(range(len(coords)), 5000)
        raca_indices = random.sample(range(len(coords)), 5000)

        # Update heatmap arrays
        for idx in mineral_indices:
            lat, lon = coords[idx]
            mineral_content = counts[idx, element_index, :, :].sum()  # Sum across the grid
            if mineral_content > 0:
                lon_idx = np.digitize(lon, lon_bins) - 1
                lat_idx = np.digitize(lat, lat_bins) - 1
                heatmap_mineral[lat_idx, lon_idx] += mineral_content

        for idx in raca_indices:
            lat, lon = coords[idx]
            raca_presence = raca_data[idx, :, :].sum()  # Sum across the grid
            if raca_presence > 0:
                lon_idx = np.digitize(lon, lon_bins) - 1
                lat_idx = np.digitize(lat, lat_bins) - 1
                heatmap_raca[lat_idx, lon_idx] += raca_presence

        # Separate normalization scales for mineral (red) and RaCA (blue) heatmaps
        norm_mineral = LogNorm(vmin=0.01, vmax=np.max(heatmap_mineral))  # Log scale for minerals to enhance visibility
        norm_raca = Normalize(vmin=np.min(heatmap_raca), vmax=np.max(heatmap_raca))  # Linear scale for RaCA

        # Plot heatmaps with independent scales
        im_mineral = ax.imshow(heatmap_mineral, cmap='Reds', alpha=0.8, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', norm=norm_mineral)
        im_raca = ax.imshow(heatmap_raca, cmap='Blues', alpha=0.8, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', norm=norm_raca)

        # Create separate colorbars for mineral and RaCA data
        divider = make_axes_locatable(ax)
        cax_mineral = divider.append_axes("right", size="5%", pad=0.05)
        cax_raca = divider.append_axes("right", size="5%", pad=0.4)

        plt.colorbar(im_mineral, cax=cax_mineral, label=f'{element_name} Intensity')
        plt.colorbar(im_raca, cax=cax_raca, label='RaCA Intensity')

    else:
        # Get random indices for minerals
        mineral_indices = random.sample(range(len(coords)), 5000)
        raca_indices = random.sample(range(len(coords)), 5000)

        # Plot random mineral data
        for idx in mineral_indices:
            lat, lon = coords[idx]
            mineral_content = counts[idx, element_index, :, :].sum()  # Sum across the grid
            if mineral_content > 0:  # Only plot locations with non-zero content
                ax.scatter(lon, lat, s=mineral_content / 10, color='red', alpha=0.6, label=f'{element_name} Site' if idx == mineral_indices[0] else "")
                labels.append(f'{element_name} Site')

        # Plot random RaCA site locations
        for idx in raca_indices:
            lat, lon = coords[idx]
            raca_presence = raca_data[idx, :, :].sum()  # Sum across the grid
            if raca_presence > 0:
                ax.scatter(lon, lat, s=raca_presence / 10, color='blue', alpha=0.6, marker='x', label=f'RaCA Site' if idx == raca_indices[0] else "")
                labels.append('RaCA Site')

    # Add legend if there are labels
    if labels:
        ax.legend(loc='upper right')

    # Title and Labels
    ax.set_title(f'{"Heatmap" if heatmap else "Scatter Plot"} for {element_name} and RaCA Sites')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Save the figure
    plt.savefig(output_path)
    plt.close()

# Create coverage maps for each element
for index, element in zip(element_indices, elements):
    output_path = f'/home/sujaynair/MRDS_Project/coverage_maps/{element}_RaCA_{"heatmap" if args.heatmap else "scatter"}_random.png'
    create_coverage_map(index, element, coords, counts, raca_data, us_shape, output_path, heatmap=args.heatmap)
