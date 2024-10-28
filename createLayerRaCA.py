import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import os
import matplotlib.pyplot as plt

# File paths provided by you
file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagridsFILLED.h5'

# Check if CUDA is available for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data from file1
print("Loading RaCA data...")
full = pd.read_pickle(file1)

# Ensure necessary columns exist
required_columns = ['rcasiteid_x', 'rcapid', 'sample_id', 'lat', 'long']
missing_columns = [col for col in required_columns if col not in full.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")

# Define identifiers
full['site_id'] = full['rcasiteid_x']  # Site ID
full['pedon_id'] = full['rcapid']      # Pedon ID
full['scan_id'] = full['sample_id']    # Scan ID

# Define columns
coordinate_columns = ['lat', 'long']
identifier_columns = ['site_id', 'pedon_id', 'scan_id']

# Scan data and lab data columns
wavelength_cols = [str(w) for w in range(365, 2501)]
lab_data_cols = ['adod', 'c_tot_ncs', 'caco3', 'n_tot_ncs', 's_tot_ncs', 'soc']

# Check for missing lab data columns
missing_lab_cols = [col for col in lab_data_cols if col not in full.columns]
if missing_lab_cols:
    print(f"The following lab data columns are missing and will be excluded: {missing_lab_cols}")
    lab_data_cols = [col for col in lab_data_cols if col in full.columns]

# Combine scan data and lab data columns
scan_lab_cols = wavelength_cols + lab_data_cols

# Filter data
print("Filtering data...")
full_filtered = full[coordinate_columns + identifier_columns + scan_lab_cols].copy()

# Handle missing data using .loc to avoid SettingWithCopyWarning
print("Handling missing data...")
full_filtered.loc[:, scan_lab_cols] = full_filtered[scan_lab_cols].fillna(0)

# Ensure the coordinate data is numeric and handle potential non-numeric entries
print("Ensuring numeric coordinates...")
full_filtered.loc[:, 'lat'] = pd.to_numeric(full_filtered['lat'], errors='coerce')
full_filtered.loc[:, 'long'] = pd.to_numeric(full_filtered['long'], errors='coerce')

# Drop rows with NaN coordinates
full_filtered = full_filtered.dropna(subset=['lat', 'long'])
print(f"Data after cleaning: {len(full_filtered)} records")

# Create a DataFrame with unique sites and their coordinates
sites = full_filtered[['site_id', 'lat', 'long']].drop_duplicates()
print(f"Total number of unique sites: {len(sites)}")

# Load 'coords' array from 'mineral_file'
print("Loading coordinates of squares...")
with h5py.File(mineral_file, 'r') as f:
    coords = f['coordinates'][:]

# Determine the correct coordinate order
# From your previous output, it seems coords[:, 0] is latitude and coords[:, 1] is longitude
coords_lat = coords[:, 0]
coords_lon = coords[:, 1]

# Verify coordinate ranges to ensure consistency
print("\nCoordinate Ranges:")
print(f"Coords latitude range: {coords_lat.min()} to {coords_lat.max()}")
print(f"Coords longitude range: {coords_lon.min()} to {coords_lon.max()}")
print(f"RaCA data latitude range: {sites['lat'].min()} to {sites['lat'].max()}")
print(f"RaCA data longitude range: {sites['long'].min()} to {sites['long'].max()}")

# Degree offset corresponding to 50 miles
miles_per_degree = 69.0  # Approximate value
degree_offset = 50 / miles_per_degree  # Degrees corresponding to 50 miles

# Initialize list to store indices of squares with data
squares_with_data_indices = []

print("\nIdentifying squares containing RaCA data...")
# Loop over all squares to find those containing data
for idx in range(len(coords)):
    lat_min = coords[idx, 0]
    lon_min = coords[idx, 1]
    lat_max = lat_min + degree_offset
    lon_max = lon_min + degree_offset

    # Filter sites within this square
    sites_in_square = sites[
        (sites['lat'] >= lat_min) & (sites['lat'] < lat_max) &
        (sites['long'] >= lon_min) & (sites['long'] < lon_max)
    ]

    if len(sites_in_square) > 0:
        squares_with_data_indices.append(idx)

print(f"Total number of squares: {len(coords)}")
print(f"Number of squares containing RaCA data: {len(squares_with_data_indices)}")

# Define convolutional models
class Scan1DConv(nn.Module):
    def __init__(self, num_wavelengths, num_lab_features, output_size=(8, 8)):
        super(Scan1DConv, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        conv_output_length = num_wavelengths // 2
        self.fc = nn.Linear(64 * conv_output_length + num_lab_features, output_size[0] * output_size[1])
        self.output_size = output_size

    def forward(self, x_scan, x_lab):
        x = self.conv1d(x_scan)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_lab], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.output_size[0], self.output_size[1])
        return x

class Conv2DReduction(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(8, 8)):
        super(Conv2DReduction, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=output_size)
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x

# Instantiate models and move them to the appropriate device
num_wavelengths = len(wavelength_cols)
num_lab_features = len(lab_data_cols)
output_size = (8, 8)

print("\nInitializing convolutional models...")
scan_conv = Scan1DConv(num_wavelengths, num_lab_features, output_size=output_size).to(device)
pedon_conv = Conv2DReduction(in_channels=1, out_channels=32, output_size=output_size).to(device)
site_conv = Conv2DReduction(in_channels=32, out_channels=64, output_size=output_size).to(device)
pixel_conv = Conv2DReduction(in_channels=64, out_channels=64, output_size=(1, 1)).to(device)  # Reduced spatial dimensions

# Set models to evaluation mode (since we're not training)
scan_conv.eval()
pedon_conv.eval()
site_conv.eval()
pixel_conv.eval()

# Initialize the aggregated output array with zeros
total_squares = len(coords)
pixel_grid_size = 50  # 50x50 grid
final_feature_dim = 64  # Reduced from 4096 to 64

# Pre-allocate the aggregated output array
# Shape: (10000, 50, 50, 64)
aggregated_output = np.zeros((total_squares, pixel_grid_size, pixel_grid_size, final_feature_dim), dtype=np.float32)

print("\nStarting processing of all squares...")
# Process all squares
for square_idx in squares_with_data_indices:
    lat_min = coords_lat[square_idx]
    lon_min = coords_lon[square_idx]
    lat_max = lat_min + degree_offset
    lon_max = lon_min + degree_offset

    # Calculate the square's index in the aggregated output array
    # Assuming squares are processed in the order of squares_with_data_indices
    processed_square_idx = squares_with_data_indices.index(square_idx)

    print(f"\nProcessing square {square_idx} with bounds:")
    print(f"Latitude: {lat_min} to {lat_max}")
    print(f"Longitude: {lon_min} to {lon_max}")

    # Filter data to only include points within this 50x50-mile square
    square_data = full_filtered[
        (full_filtered['lat'] >= lat_min) & (full_filtered['lat'] < lat_max) &
        (full_filtered['long'] >= lon_min) & (full_filtered['long'] < lon_max)
    ]

    print(f"Total number of records within square {square_idx}: {len(square_data)}")

    if len(square_data) == 0:
        print(f"No data found within square {square_idx}. Skipping.")
        continue

    # Define the 1x1-mile pixel grid within the square
    lat_bins = np.linspace(lat_min, lat_max, pixel_grid_size + 1)
    lon_bins = np.linspace(lon_min, lon_max, pixel_grid_size + 1)

    # Loop over each 1x1-mile pixel within the square
    for i in range(pixel_grid_size):
        for j in range(pixel_grid_size):
            lat_lower = lat_bins[i]
            lat_upper = lat_bins[i + 1]
            lon_lower = lon_bins[j]
            lon_upper = lon_bins[j + 1]

            # Filter data within this pixel
            pixel_data = square_data[
                (square_data['lat'] >= lat_lower) & (square_data['lat'] < lat_upper) &
                (square_data['long'] >= lon_lower) & (square_data['long'] < lon_upper)
            ]

            if len(pixel_data) == 0:
                # No data in this pixel; the aggregated_output remains zero
                continue

            # Initialize pixel sum
            pixel_sum = None

            # Process each site within the pixel
            sites_in_pixel = pixel_data['site_id'].unique()
            for site_id in sites_in_pixel:
                site_data = pixel_data[pixel_data['site_id'] == site_id]
                pedons_in_site = site_data['pedon_id'].unique()

                site_sum = None

                # Process each pedon within the site
                for pedon_id in pedons_in_site:
                    pedon_data = site_data[site_data['pedon_id'] == pedon_id]
                    scans_in_pedon = pedon_data['scan_id'].unique()

                    pedon_sum = None

                    # Process each scan within the pedon
                    for scan_id in scans_in_pedon:
                        scan_row = pedon_data[pedon_data['scan_id'] == scan_id]

                        scan_spectral = scan_row[wavelength_cols].values.astype(np.float32)
                        scan_lab = scan_row[lab_data_cols].values.astype(np.float32)

                        # Convert to tensors and move to device
                        scan_spectral_tensor = torch.tensor(scan_spectral).to(device)
                        scan_lab_tensor = torch.tensor(scan_lab).to(device)

                        # Ensure the tensors have the correct shape
                        if scan_spectral_tensor.ndim == 1:
                            scan_spectral_tensor = scan_spectral_tensor.unsqueeze(0)
                        scan_spectral_tensor = scan_spectral_tensor.unsqueeze(1)  # Shape: [1, 1, num_wavelengths]

                        if scan_lab_tensor.ndim == 1:
                            scan_lab_tensor = scan_lab_tensor.unsqueeze(0)  # Shape: [1, num_lab_features]

                        # Apply 1D convolution
                        with torch.no_grad():
                            scan_output = scan_conv(scan_spectral_tensor, scan_lab_tensor)  # Output shape: [1, 8, 8]

                        if pedon_sum is None:
                            pedon_sum = scan_output
                        else:
                            pedon_sum += scan_output

                    if pedon_sum is not None:
                        # Add channel dimension for Conv2D
                        pedon_sum = pedon_sum.unsqueeze(1)  # Shape: [1, 1, 8, 8]
                        with torch.no_grad():
                            pedon_output = pedon_conv(pedon_sum)  # Output shape: [1, 32, 8, 8]
                        if site_sum is None:
                            site_sum = pedon_output
                        else:
                            site_sum += pedon_output

            if site_sum is not None:
                with torch.no_grad():
                    site_output = site_conv(site_sum)  # Output shape: [1, 64, 8, 8]
                if pixel_sum is None:
                    pixel_sum = site_output
                else:
                    pixel_sum += site_output

            if pixel_sum is not None:
                with torch.no_grad():
                    pixel_output = pixel_conv(pixel_sum)  # Output shape: [1, 64, 1, 1]
                # Flatten to [1, 64]
                pixel_features = pixel_output.view(pixel_output.size(0), -1).cpu().numpy()  # Shape: [1, 64]
                # Assign to the aggregated output array
                aggregated_output[square_idx, i, j, :] = pixel_features[0]
                print(f"Processed pixel ({i+1}/{pixel_grid_size}, {j+1}/{pixel_grid_size}) in square {square_idx}")
            else:
                # Pixel remains as zero vector
                pass

    print("\nProcessing complete. Saving results...")

    # Save the aggregated output to the specified HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('aggregated_output', data=aggregated_output)

    print(f"Aggregated output saved to: {output_file}")
    print(f"Aggregated Output Shape: {aggregated_output.shape}")
    print(f"Aggregated Output Data Type: {aggregated_output.dtype}")


# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import pdb

# # Path to the aggregated output file
# output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagridsFILLED.h5'

# # Load the aggregated output
# with h5py.File(output_file, 'r') as f:
#     pdb.set_trace()
#     aggregated_output = f['aggregated_output'][:]

# print(f"Aggregated Output Shape: {aggregated_output.shape}")
# print(f"Aggregated Output Data Type: {aggregated_output.dtype}")

# # Function to visualize a single square
# def visualize_square(square_index):
#     square_data = aggregated_output[square_index]  # Shape: [50, 50, 64]
    
#     # Sum over the feature dimension to get a 2D representation
#     square_sum = np.sum(square_data, axis=2)  # Shape: [50, 50]
    
#     plt.figure(figsize=(8, 6))
#     plt.imshow(square_sum, cmap='viridis', interpolation='nearest')
#     plt.title(f'Square Index: {square_index} - Sum of Features')
#     plt.colorbar(label='Summed Feature Values')
#     plt.xlabel('Longitude Pixel')
#     plt.ylabel('Latitude Pixel')
#     plt.savefig(f"/home/sujaynair/MRDS_Project/prepared_data_TILES/{square_index}")

# # Visualize the first 10 squares
# num_squares_to_visualize = 3
# for i in range(num_squares_to_visualize):
#     print(f"Visualizing Square {i}...")
#     visualize_square(i)
