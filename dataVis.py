import h5py
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import numpy as np

# File paths
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
output_path = '/home/sujaynair/MRDS_Project/tilingVIS/squares_on_map.png'

# Load the shapefile
print("Loading shapefile...")
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']

# Convert miles to degrees latitude and longitude
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

# Load HDF5 file and extract coordinates
def load_h5_coordinates(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        coords = f['coordinates'][:]
    return coords

def visualize_squares_with_coords(coords, us_shape, output_path):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')
    
    total_squares = len(coords)
    
    for idx, coord in enumerate(coords):
        lat_start, lon_start = coord
        square_geom = box(lon_start, lat_start, lon_start + miles_to_degrees_lon(50, lat_start), lat_start + miles_to_degrees_lat(50))
        gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='red')
        
        if idx % 100 == 0 or idx == total_squares - 1:
            print(f"Processed {idx + 1}/{total_squares} squares...")
    
    plt.title("Squares Mapped on US")
    plt.savefig(output_path)
    plt.close(fig)

# Main function
if __name__ == "__main__":
    print("Loading HDF5 coordinates...")
    coords = load_h5_coordinates(h5_file_path)
    
    print("Visualizing squares on map...")
    visualize_squares_with_coords(coords, us_shape, output_path)
    
    print(f"Visualization saved at {output_path}")
