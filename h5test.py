import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from shapely.geometry import box

# File paths
h5_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineral_data.h5'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'

# Function to load HDF5 file and print information
def load_and_print_h5_info(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        print("HDF5 file structure:")
        f.visit(print)

        # Print dataset shapes
        counts_shape = f['counts'].shape
        qualities_shape = f['qualities'].shape
        print(f"\nShape of 'counts' dataset: {counts_shape}")
        print(f"Shape of 'qualities' dataset: {qualities_shape}")

        # Print dataset stats
        counts_data = f['counts'][:]
        qualities_data = f['qualities'][:]
        print(f"\nCounts dataset min: {np.min(counts_data)}, max: {np.max(counts_data)}, mean: {np.mean(counts_data)}")
        print(f"Qualities dataset min: {np.min(qualities_data)}, max: {np.max(qualities_data)}, mean: {np.mean(qualities_data)}")

        return counts_data, qualities_data

# Function to visualize the counts and qualities for a specific square
def visualize_square_data(counts_data, qualities_data, square_index, minerals, us_shape):
    fig, axs = plt.subplots(2, len(minerals), figsize=(20, 10))

    for i, mineral in enumerate(minerals):
        # Visualize counts
        counts = counts_data[square_index, i, :, :]
        axs[0, i].imshow(counts, cmap='hot', interpolation='nearest')
        axs[0, i].set_title(f"{mineral} Counts")
        axs[0, i].axis('off')

        # Visualize qualities
        qualities = qualities_data[square_index, i, :, :]
        axs[1, i].imshow(qualities, cmap='cool', interpolation='nearest')
        axs[1, i].set_title(f"{mineral} Qualities")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Function to visualize the locations of all squares on the US map
def visualize_all_squares(us_shape, squares):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')

    for square in squares:
        lat_start, lon_start = square
        square_geom = box(lon_start, lat_start, lon_start + 50, lat_start + 50)
        gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='red')

    plt.title("Locations of All Squares Overlaying the US")
    plt.show()

# Main script
if __name__ == "__main__":
    print("Loading HDF5 file and printing information...")
    counts_data, qualities_data = load_and_print_h5_info(h5_file_path)

    # Load the shapefile
    print("Loading shapefile...")
    us_shapefile = gpd.read_file(shapefile_path)
    us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']

    # Define minerals and squares (you might need to pass squares if they are not stored)
    minerals = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
    
    # Visualize data for a specific square (example: first square)
    square_index = 0
    print(f"Visualizing data for square {square_index}...")
    visualize_square_data(counts_data, qualities_data, square_index, minerals, us_shape)

    # Visualize all squares on the US map
    # You will need to recreate or load the square coordinates used in your previous script
    # Example squares coordinates, replace this with actual data
    squares = [(24.396308, -125.0), (35.0, -90.0), (40.0, -75.0)]  # Example coordinates
    print("Visualizing all squares on the US map...")
    visualize_all_squares(us_shape, squares)
