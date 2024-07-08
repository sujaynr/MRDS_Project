import h5py
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import numpy as np

# File paths
original_h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
fault_h5_file_path = 'prepared_data_TILES/faultData.h5'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
fault_shapefile_path = '/home/sujaynair/GDB/SHP/Qfaults_US_Database.shp'
output_path = '/home/sujaynair/MRDS_Project/tilingVIS/squares_with_faults_on_map.png'

# Load the shapefile
print("Loading shapefile...")
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']

# Load fault lines shapefile
print("Loading fault lines shapefile...")
fault_lines = gpd.read_file(fault_shapefile_path)

# Convert miles to degrees latitude and longitude
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

# Load HDF5 file and extract coordinates
def load_h5_coordinates(h5_file_path, sample_fraction=0.05):
    with h5py.File(h5_file_path, 'r') as f:
        coords = f['coordinates'][:]
    # Sample a fraction of the data
    num_samples = int(len(coords) * sample_fraction)
    return coords[:num_samples]

def save_fault_dataset(fault_dataset, fault_h5_file_path):
    with h5py.File(fault_h5_file_path, 'w') as f:
        f.create_dataset('faults', data=fault_dataset)
    print(f"Fault dataset saved at {fault_h5_file_path}")

def visualize_squares_with_faults(coords, fault_lines, us_shape, output_path, minerals=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')

    for idx, coord in enumerate(coords):
        lat_start, lon_start = coord
        square_geom = box(lon_start, lat_start, lon_start + miles_to_degrees_lon(50, lat_start), lat_start + miles_to_degrees_lat(50))
        
        # Check if the square intersects with fault lines
        cell_box = box(lon_start, lat_start, lon_start + miles_to_degrees_lon(50, lat_start), lat_start + miles_to_degrees_lat(50))
        intersecting_faults = fault_lines[fault_lines.intersects(cell_box)]
        
        if not intersecting_faults.empty:
            gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='yellow')  # Color intersecting squares black
        else:
            gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='red')  # Color non-intersecting squares red

    # Add circle points for squares 0-9 from the mineral dataset
    if minerals is not None:
        for idx in range(min(10, len(minerals))):  # Ensure we only plot up to the first 10 squares
            mineral_coord = minerals[idx]
            lat_start, lon_start = mineral_coord
            ax.scatter(lon_start, lat_start, color='green', marker='o', s=50)  # Circle point for minerals

    fault_lines.boundary.plot(ax=ax, color='blue')
    
    plt.title("Squares and Fault Lines on US Map")
    plt.xlim([-125, -66])
    plt.ylim([24, 50])
    plt.savefig(output_path)
    plt.close(fig)

def create_fault_dataset(coords, fault_lines, num_minerals=10, grid_size=50, cell_size=1):
    num_squares = len(coords)
    slip_rate_values = {
        'Less than 0.2 mm/yr': 0.1,
        'Between 1.0 and 5.0 mm/yr': 3.0,
        'Greater than 5.0 mm/yr': 5.0,
        'Between 0.2 and 1.0 mm/yr': 0.6,
        '0.2 +/- 0.1 mm/yr': 0.2,
        'Insufficient data': np.nan,
        'Unspecified': np.nan,
        None: np.nan
    }

    # Original dimensions of fault dataset
    original_fault_dataset = np.full((num_squares, grid_size, grid_size), np.nan)

    for idx, coord in enumerate(coords):
        lat_start, lon_start = coord
        for i in range(grid_size):
            for j in range(grid_size):
                lat_min = lat_start + i * miles_to_degrees_lat(cell_size)
                lat_max = lat_start + (i + 1) * miles_to_degrees_lat(cell_size)
                lon_min = lon_start + j * miles_to_degrees_lon(cell_size, lat_start)
                lon_max = lon_start + (j + 1) * miles_to_degrees_lon(cell_size, lat_start)

                cell_box = box(lon_min, lat_min, lon_max, lat_max)
                intersecting_faults = fault_lines[fault_lines.intersects(cell_box)]

                if not intersecting_faults.empty:
                    slip_rates = intersecting_faults['slip_rate'].map(slip_rate_values)
                    if not slip_rates.isna().all():
                        original_fault_dataset[idx, i, j] = np.nanmean(slip_rates)

        # Progress update
        print(f"Processed square {idx + 1}/{num_squares}")

    # Extend to include 10 minerals
    fault_dataset = np.repeat(original_fault_dataset[:, np.newaxis, :, :], num_minerals, axis=1)

    return fault_dataset

# Assuming you have the other functions (like load_h5_coordinates, save_fault_dataset) unchanged

# Main function
if __name__ == "__main__":
    print("Loading HDF5 coordinates...")
    coords = load_h5_coordinates(original_h5_file_path, sample_fraction=1)
    
    print("Creating fault dataset...")
    fault_dataset = create_fault_dataset(coords, fault_lines, num_minerals=10)
    
    print("Saving fault dataset...")
    save_fault_dataset(fault_dataset, fault_h5_file_path)
    
    print("Visualizing squares with faults on map...")
    visualize_squares_with_faults(coords, fault_lines, us_shape, output_path, minerals=coords)
    
    print(f"Visualization saved at {output_path}")
