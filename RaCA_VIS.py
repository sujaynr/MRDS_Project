import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
# Define file paths
file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
file2 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDatasetAux_AA_v1.pkl"
mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/processed_cells.h5'
plot_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/average_wavelength_intensity.png'

# Load data from pickle files
full = pd.read_pickle(file1)
aux = pd.read_pickle(file2)
pdb.set_trace()
# Load coordinates from the h5 file
with h5py.File(mineral_file, 'r') as f:
    coords = f['coordinates'][:]

# Get the list of wavelength columns
wavelength_cols = [str(i) for i in range(365, 2501)]
num_wavelengths = len(wavelength_cols)
num_cells_per_square = 50 * 50  # 2500

# Function to calculate new latitude and longitude given a distance in miles
def calculate_lat_long(lat, long, miles_lat, miles_long):
    # Convert miles to degrees (approximate conversion)
    miles_to_deg = 1 / 69  # 1 degree latitude is approximately 69 miles
    new_lat = lat + (miles_lat * miles_to_deg)
    new_long = long + (miles_long * miles_to_deg / np.cos(np.deg2rad(lat)))
    return new_lat, new_long

# Initialize a 3D numpy array to store the results
num_squares = 10000
results = np.zeros((num_squares, num_cells_per_square, num_wavelengths))

# Loop through the first 5 squares
for square_index, (square_lat, square_long) in enumerate(coords[:num_squares]):
    print(f"\nProcessing square {square_index + 1}")
    for i in range(50):
        for j in range(50):
            cell_index = i * 50 + j
            cell_lat, cell_long = calculate_lat_long(square_lat, square_long, i, j)
            # Find all scans within this cell
            cell_scans = full[(full['lat'].between(cell_lat, cell_lat + 1/69)) & 
                              (full['long'].between(cell_long, cell_long + 1/(69 * np.cos(np.deg2rad(cell_lat)))))]
            
            if not cell_scans.empty:
                # Average the wavelength columns
                avg_vector = cell_scans[wavelength_cols].mean().values
                results[square_index, cell_index, :] = avg_vector
            # If no scans, the cell remains zeroed
            
            print(f"Processed cell ({i}, {j}) in square {square_index + 1} with {len(cell_scans)} scans.")

# Save the processed data to an HDF5 file
with h5py.File(output_file, 'w') as f:
    f.create_dataset('processed_data', data=results)
    pdb.set_trace()
print(f"\nProcessed data saved to {output_file}")

# Compute the mean intensity across all squares and cells
mean_intensity = np.mean(results, axis=(0, 1))

# Plotting the result
plt.figure(figsize=(12, 8))
sns.heatmap(mean_intensity.reshape(1, -1), cmap='viridis', cbar_kws={'label': 'Average Wavelength Intensity'})
plt.title('Heatmap of Averaged Wavelength Intensity')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")
plt.show()
