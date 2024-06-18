import os
import h5py

import matplotlib.pyplot as plt

h5_file_path = '/Users/sujaynair/Documents/MRDS_Project/prepared_data_TILES/mineral_data.h5'
output_dir = '/Users/sujaynair/Documents/MRDS_Project/tilingVIS'

# Function to recursively print the structure of the HDF5 file
def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"{name} - Dataset with shape {obj.shape} and dtype {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{name} - Group with {len(obj)} members")

# Load the HDF5 file
with h5py.File(h5_file_path, 'r') as data:
    data.visititems(print_structure)
    
    # Access the 'counts' dataset
    counts = data['counts']
    qualities = data['qualities']
    
    # Print details of the datasets
    print(f"\n'counts' dataset shape: {counts.shape}, dtype: {counts.dtype}")
    print(f"'qualities' dataset shape: {qualities.shape}, dtype: {qualities.dtype}")
    
    # Visualize all 20 slices of mineral 1
    num_slices = 20  # Number of slices to visualize
    mineral_index = 0  # Index of the mineral to visualize

    for i in range(num_slices):
        # Create a new figure for each slice
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the counts
        im1 = axs[0].imshow(counts[i, mineral_index, :, :], cmap='viridis')
        axs[0].set_title(f'Counts - Slice {i+1}, Mineral 1')
        fig.colorbar(im1, ax=axs[0])
        
        # Plot the qualities
        im2 = axs[1].imshow(qualities[i, mineral_index, :, :], cmap='plasma')
        axs[1].set_title(f'Qualities - Slice {i+1}, Mineral 1')
        fig.colorbar(im2, ax=axs[1])
    
        plt.tight_layout()
        
        # Save the figure in the output directory
        output_path = os.path.join(output_dir, f'slice_{i+1}.png')
        plt.savefig(output_path)
        
        plt.show()
