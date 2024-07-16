import requests
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

# URL for the USGS Elevation Point Query Service
url = "https://epqs.nationalmap.gov/v1/json"
mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/elevation_data.h5'

with h5py.File(mineral_file, 'r') as f:
    coords = f['coordinates'][:]

def get_elevation(lat, lon):
    try:
        params = {
            'x': lon,
            'y': lat,
            'units': 'Meters',
            'format': 'json',
            'wkid': 4326,
            'includeDate': 'false'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        
        elevation = data['value']
        return float(elevation)
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return np.nan  # Return NaN for request failures
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error parsing elevation data: {e}")
        return np.nan  # Return NaN for parsing errors

# Conversion factor (degrees per mile)
miles_per_degree_lat = 69.0
miles_per_degree_lon = 69.172

# Function to calculate latitude and longitude offsets
def calculate_offset(lat, miles_lat, miles_lon):
    lat_offset = miles_lat / miles_per_degree_lat
    lon_offset = miles_lon / (miles_per_degree_lon * math.cos(math.radians(lat)))
    return lat_offset, lon_offset

# Initialize the result array
elevation_data = np.zeros((10000, 50, 50))

# Loop through each square
for square_idx in range(10000):
    bottom_left_lat = coords[square_idx][0]
    bottom_left_lon = coords[square_idx][1]

    for i in range(50):
        for j in range(50):
            print(f"Processing cell {square_idx * 50 * 50 + i * 50 + j + 1} out of 2500")
            # Calculate the center of each cell
            cell_center_lat = bottom_left_lat + (i + 0.5) / miles_per_degree_lat
            _, lon_offset = calculate_offset(cell_center_lat, 0, (j + 0.5))
            cell_center_lon = bottom_left_lon + lon_offset

            # Get the elevation for the cell center
            elevation = get_elevation(cell_center_lat, cell_center_lon)
            elevation_data[square_idx, i, j] = elevation

    print(f"Processed square {square_idx + 1} out of 10000")

# Save the elevation data to an HDF5 file
with h5py.File(output_file, 'w') as f:
    f.create_dataset('elevation_data', data=elevation_data)

print("Elevation data processing complete.")
