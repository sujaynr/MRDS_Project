import pdb
import h5py
import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
fault_line_file = '/home/sujaynair/GDB/SHP/Qfaults_US_Database.shp'
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
output_h5_file_path = 'prepared_data_TILES/dataCoordsFaults.h5'

# Load fault line data
print("Loading fault line data...")
fault_lines = gpd.read_file(fault_line_file)

# Convert miles to degrees latitude and longitude
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

# Convert slip rate descriptions to numerical values
def convert_slip_rate(slip_rate):
    if pd.isnull(slip_rate) or slip_rate in ['Unspecified', 'Insufficient data', 'None']:
        return 0
    if 'Less than 0.2 mm/yr' in slip_rate:
        return 0.1
    if '0.2 +/- 0.1 mm/yr' in slip_rate:
        return 0.2
    if 'Between 0.2 and 1.0 mm/yr' in slip_rate:
        return 0.6
    if 'Between 1.0 and 5.0 mm/yr' in slip_rate:
        return 3.0
    if 'Greater than 5.0 mm/yr' in slip_rate:
        return 5.0
    return 0

# Apply conversion to the slip_rate column
fault_lines['slip_rate_numeric'] = fault_lines['slip_rate'].apply(convert_slip_rate)

# Load HDF5 file and extract coordinates
def load_h5_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        
        coords = f['coordinates'][:]
        counts = f['counts'][:]
        qualities = f['qualities'][:]
        pdb.set_trace()
    return coords, counts, qualities

# Check if cell intersects with fault lines and get slip rates
def check_fault_intersection(lat_start, lon_start, grid_size=50, cell_size=1):
    lat_length = miles_to_degrees_lat(cell_size)
    lon_length = miles_to_degrees_lon(cell_size, lat_start)
    faults_presence = np.zeros((grid_size, grid_size))
    faults_slip_rate = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            cell_bounds = box(lon_start + j * lon_length, lat_start + i * lat_length,
                              lon_start + (j + 1) * lon_length, lat_start + (i + 1) * lat_length)
            faults_in_cell = fault_lines[fault_lines.intersects(cell_bounds)]
            if not faults_in_cell.empty:
                faults_presence[i, j] = 1
                faults_slip_rate[i, j] = faults_in_cell['slip_rate_numeric'].mean()

    return faults_presence, faults_slip_rate

# Main function to update HDF5 file with fault data
def update_hdf5_with_faults(h5_file_path, output_h5_file_path, coords):
    with h5py.File(h5_file_path, 'r') as f:
        counts = f['counts'][:]
        qualities = f['qualities'][:]
    
    num_squares, num_minerals, grid_size, _ = counts.shape

    with h5py.File(output_h5_file_path, 'w') as f:
        count_ds = f.create_dataset('counts', (num_squares, num_minerals, grid_size, grid_size), dtype='f')
        quality_ds = f.create_dataset('qualities', (num_squares, num_minerals, grid_size, grid_size), dtype='f')
        fault_presence_ds = f.create_dataset('fault_presence', (num_squares, grid_size, grid_size), dtype='f')
        fault_slip_rate_ds = f.create_dataset('fault_slip_rate', (num_squares, grid_size, grid_size), dtype='f')
        coord_ds = f.create_dataset('coordinates', data=coords)
        
        for idx, (lat_start, lon_start) in enumerate(coords):
            print(f"Processing square {idx + 1}/{num_squares}...")
            faults_presence, faults_slip_rate = check_fault_intersection(lat_start, lon_start, grid_size)
            count_ds[idx] = counts[idx]
            quality_ds[idx] = qualities[idx]
            fault_presence_ds[idx] = faults_presence
            fault_slip_rate_ds[idx] = faults_slip_rate

    print(f"Updated HDF5 file saved at {output_h5_file_path}")

# Visualization of squares and fault lines
def visualize_squares_and_faults(squares, fault_lines, us_shape, output_path='generated_squares_with_faults.png'):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')

    for square in squares:
        lat_start, lon_start = square
        square_geom = box(lon_start, lat_start, lon_start + miles_to_degrees_lon(50, lat_start), lat_start + miles_to_degrees_lat(50))
        gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='red')
    
    fault_lines.plot(ax=ax, color='blue', linewidth=0.5)
    
    # Set plot limits to the US boundaries
    ax.set_xlim(-125, -66.93457)
    ax.set_ylim(24.396308, 49.384358)
    
    plt.title("Randomly Generated Squares and Fault Lines Overlaying the US")
    plt.savefig(output_path)
    plt.close(fig)

# Main script
if __name__ == "__main__":
    print("Loading HDF5 data...")
    coords, counts, qualities = load_h5_data(h5_file_path)
    
    print("Updating HDF5 file with fault data...")
    # update_hdf5_with_faults(h5_file_path, output_h5_file_path, coords)

    print("Data update completed.")

    print("Visualizing squares and fault lines...")
    us_shapefile = gpd.read_file(shapefile_path)
    us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']
    visualize_squares_and_faults(coords, fault_lines, us_shape, output_path='/home/sujaynair/MRDS_Project/tilingVIS/generated_squares_with_faults.png')
