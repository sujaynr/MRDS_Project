import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Haversine function to calculate bird's-eye distance between two lat/long points
def haversine(lon1, lat1, lon2, lat2):
    R = 3958.8  # Radius of Earth in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load reflectance data
reflectance_df = pd.read_pickle("/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl")

# Load coordinates and mineral data
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
with h5py.File(h5_file_path, 'r') as f:
    coords = f['coordinates'][:]
    counts = f['counts'][:]

# Select the mineral index (e.g., 0 for Nickel)
layer_names = [
    "Gold", "Silver", "Zinc", "Lead", "Copper", 
    "Nickel", "Iron", "Uranium", "Tungsten", "Manganese",
    "Faults", "GeoAge Min", "GeoAge Max", 
    "Elevation", "RaCA Data"
]

# Number of random squares to select
num_squares = min(50, len(coords))

# Randomly sample the square indices
random_indices = np.random.choice(len(coords), num_squares, replace=False)

# Initialize a list to store R² scores across all sites
all_r2_scores = []

for mineral_index in range(10):
    mineral = layer_names[mineral_index]
    
    # Iterate over the selected random grid squares
    for i in random_indices:
        coord = coords[i]
        
        # Check if the current square has any of the selected mineral
        if counts[i, mineral_index, :, :].sum() > 0:
            print(f"Processing square {i+1}/{len(coords)} with mineral occurrences")

            # Iterate over each cell in the 50x50 grid within the square
            for row in range(50):
                for col in range(50):
                    if counts[i, mineral_index, row, col] > 0:  # If mineral is present in this cell
                        # Calculate the lat/long of the cell within the square
                        lat = coord[0] + (row / 50) * 50 / 69.0  # Approximate lat degree change per mile
                        lon = coord[1] + (col / 50) * 50 / (69.0 * np.cos(np.radians(lat)))  # Approximate lon degree change per mile

                        # Calculate distances from each RaCA site to the current mineral occurrence
                        reflectance_df['distance_to_resource'] = reflectance_df.apply(
                            lambda row: haversine(lon, lat, row['long'], row['lat']), axis=1)

                        # Drop rows with NaN values in the distance
                        reflectance_df = reflectance_df.dropna(subset=['distance_to_resource'])

                        # Initialize a list to store R² values for the current occurrence
                        r2_scores = []

                        for wavelength in range(365, 2501):
                            # Prepare data for linear regression
                            X = reflectance_df['distance_to_resource'].values.reshape(-1, 1)
                            y = reflectance_df[str(wavelength)].values

                            # Remove NaN values
                            mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
                            X, y = X[mask], y[mask]

                            # Perform linear regression
                            if len(X) > 0:
                                model = LinearRegression().fit(X, y)
                                predictions = model.predict(X)
                                r2 = r2_score(y, predictions)
                            else:
                                r2 = np.nan

                            r2_scores.append(r2)

                        # Store the R² values for the current cell within the square
                        all_r2_scores.append(r2_scores)

            print(f"Finished processing square {i+1}/{len(coords)}")

    # After processing all squares, calculate the average R² values across all occurrences
    average_r2_scores = np.nanmean(all_r2_scores, axis=0)
    print("Finished averaging R² scores across all occurrences.")

    # Plot R^2 values as a function of wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(range(365, 2501), average_r2_scores, label=f'Average R^2 vs Wavelength ({mineral})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('R^2')
    plt.title(f'Average R^2 of Reflectance vs. Distance to {mineral} Resources Across All Occurrences')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Average_R2_vs_Wavelength_{mineral}.png')
    plt.show()
    print(f"Plot saved as 'Average_R2_vs_Wavelength_{mineral}.png'.")
