import pdb
import h5py
import pandas as pd
import numpy as np
import os

# # Load data from pickle files
# file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
# file2 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDatasetAux_AA_v1.pkl"
# mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagrids.h5'

# full = pd.read_pickle(file1)
# aux = pd.read_pickle(file2)

# # Load coordinates from HDF5 file
# with h5py.File(mineral_file, 'r') as f:
#     coords = f['coordinates'][:]

# # Function to create a grid and count RaCA sites within each cell
# def create_grid_and_count_sites(coords, sites, n):
#     """
#     Creates a 50x50 grid for the n-th square and counts how many RaCA sites fall within each cell.
    
#     Parameters:
#     coords (np.array): An array of shape (10k, 2) with lat/long of the bottom-left points of the squares.
#     sites (pd.DataFrame): A DataFrame containing the site data with columns including 'rcasiteid_y', 'lat', and 'long'.
#     n (int): The index of the square to check.
    
#     Returns:
#     np.array: A 50x50 grid with the count of RaCA sites in each cell.
#     """
#     # Define the size of the square in degrees (approximation, as 1 degree latitude ~ 69 miles)
#     square_size_in_degrees = 50 / 69.0
    
#     # Get the bottom-left corner of the n-th square
#     bottom_left_lat = coords[n, 0]
#     bottom_left_long = coords[n, 1]
    
#     # Calculate the top-right corner of the square
#     top_right_lat = bottom_left_lat + square_size_in_degrees
#     top_right_long = bottom_left_long + square_size_in_degrees
    
#     # Create an empty 50x50 grid
#     grid = np.zeros((50, 50), dtype=int)
    
#     # Find sites within the bounds of the square
#     sites_in_square = sites[
#         (sites['lat'] >= bottom_left_lat) &
#         (sites['lat'] <= top_right_lat) &
#         (sites['long'] >= bottom_left_long) &
#         (sites['long'] <= top_right_long)
#     ]
    
#     # Calculate row and column indices for each site
#     rows = ((top_right_lat - sites_in_square['lat']) // (square_size_in_degrees / 50)).astype(int)
#     cols = ((sites_in_square['long'] - bottom_left_long) // (square_size_in_degrees / 50)).astype(int)
    
#     # Count sites within each cell
#     for row, col in zip(rows, cols):
#         if 0 <= row < 50 and 0 <= col < 50:  # Ensure indices are within grid bounds
#             grid[row, col] += 1
    
#     return grid

# # Create an HDF5 file to store all grids
# with h5py.File(output_file, 'w') as h5f:
#     # Create a dataset to store the 10k grids
#     dset = h5f.create_dataset('racagrids', shape=(10000, 50, 50), dtype=int)
    
#     # Iterate through all 10,000 coordinates and create the grids
#     for n in range(len(coords)):
#         if {'lat', 'long', 'rcasiteid_y'}.issubset(full.columns):
#             grid = create_grid_and_count_sites(coords, full, n)
#             dset[n] = grid  # Save each grid in the HDF5 dataset
#             print(f"Processed square {n + 1}")
#         else:
#             print("Required columns are not available in the dataset.")
#             break

# print(f"All 10k grids have been saved in {output_file}")

# Open the HDF5 file containing the grids
with h5py.File(output_file, 'r') as h5f:
    # Read the dataset containing the grids
    grids_dataset = h5f['racagrids']
    pdb.set_trace()
    
    # Access and print information about the dataset
    print(f"Dataset shape: {grids_dataset.shape}")
    print(f"Dataset dtype: {grids_dataset.dtype}")
    
    # Access and print a specific grid
    grid_index = 0  # Index of the grid to access
    grid = grids_dataset[grid_index]
    print(f"Grid at index {grid_index}:")
    print(grid)