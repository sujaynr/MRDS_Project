import pandas as pd
import numpy as np
import h5py
import os
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt

# File paths
file_path = '/home/sujaynair/mrds.csv'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
data_dir = 'prepared_data_TILES'
h5_file_path = os.path.join(data_dir, '100kmineralDataWithCoords.h5')

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

print("Loading shapefile...")
# Load the shapefile
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']

# Function to check if a point is within the US boundary
def is_within_us(lat, lon, us_shape):
    point = Point(lon, lat)
    return us_shape.contains(point).any()

# Convert miles to degrees latitude and longitude
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

def is_square_within_us(lat_start, lon_start, us_shape, side_length_miles=50):
    lat_length = miles_to_degrees_lat(side_length_miles)
    lon_length = miles_to_degrees_lon(side_length_miles, lat_start)
    points = [
        Point(lon_start, lat_start),
        Point(lon_start + lon_length, lat_start),
        Point(lon_start, lat_start + lat_length),
        Point(lon_start + lon_length, lat_start + lat_length)
    ]
    return all(us_shape.contains(point).any() for point in points)

print("Loading and processing the mineral dataset...")
# Load the mineral dataset
df = pd.read_csv(file_path)
df['region'] = df['region'].fillna('NA')
values = {"commod1": "", "commod2": "", "commod3": ""}
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].fillna(value=values)
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].astype(str)
df['commodities'] = df.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
df = df.assign(commodities=df['commodities'].str.split(',')).explode('commodities')
df = df[df['commodities'] != '']
df = df[df['dev_stat'] != 'Plant']

# Filter for specific minerals
specific_minerals = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
df = df[df['commodities'].isin(specific_minerals)]

# Filter minerals with at least 1000 datapoints
print("Filtering minerals with at least 1000 datapoints...")
mineral_counts = df['commodities'].value_counts()
valid_minerals = mineral_counts[mineral_counts >= 1000].index
df = df[df['commodities'].isin(valid_minerals)]

# Ensure the mineral mapping is consistent with the required order
mineral_map = {mineral: idx for idx, mineral in enumerate(['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese'])}

# Generate random 50-mile by 50-mile squares within the US
def generate_random_squares(df, num_squares, us_shape):
    squares = []
    counts = []
    qualities = []
    c = 0
    while len(squares) < num_squares:
        c+=1
        lat = np.random.uniform(24.396308, 49.384358)  # US lat range
        lon = np.random.uniform(-125.0, -66.93457)  # US lon range
        if is_square_within_us(lat, lon, us_shape):
            count, quality = fill_cells(df, (lat, lon))
            if np.sum(count) > 0:  # Check if the square is non-empty
                squares.append((lat, lon))
                counts.append(count)
                qualities.append(quality)
                # Check progress:
            print(f"Generated {len(squares)} squares out of {c} total")
    return squares, counts, qualities

# Function to fill in the cells for each square
def fill_cells(df, square, grid_size=50, cell_size=1):
    lat_start, lon_start = square
    lat_end = lat_start + (grid_size / cell_size) * miles_to_degrees_lat(cell_size)
    lon_end = lon_start + (grid_size / cell_size) * miles_to_degrees_lon(cell_size, lat_end)
    minerals = df['commodities'].unique()
    cell_counts = np.zeros((len(minerals), grid_size, grid_size))
    cell_qualities = np.full((len(minerals), grid_size, grid_size), np.nan)
    region_df = df[
                (df['latitude'] >= lat_start) & (df['latitude'] < lat_end) &
                (df['longitude'] >= lon_start) & (df['longitude'] < lon_end)
            ]

    for i in range(grid_size):
        for j in range(grid_size):
            lat_min = lat_start + i * miles_to_degrees_lat(cell_size)
            lat_max = lat_start + (i + 1) * miles_to_degrees_lat(cell_size)
            lon_min = lon_start + j * miles_to_degrees_lon(cell_size, lat_start)
            lon_max = lon_start + (j + 1) * miles_to_degrees_lon(cell_size, lat_start)

            cell_data = region_df[
                (region_df['latitude'] >= lat_min) & (region_df['latitude'] < lat_max) &
                (region_df['longitude'] >= lon_min) & (region_df['longitude'] < lon_max)
            ]

            for mineral in minerals:
                mineral_data = cell_data[cell_data['commodities'] == mineral]
                count = len(mineral_data)
                cell_counts[mineral_map[mineral], i, j] = count
                if count > 0:
                    cell_qualities[mineral_map[mineral], i, j] = np.mean(mineral_data['score'].map({'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))

    return cell_counts, cell_qualities

# Main function to create the HDF5 file
# Main function to create the HDF5 file
def create_hdf5_file(h5_file_path, df, num_squares, grid_size=50, cell_size=1, save_interval=20000):
    print("Generating random squares...")
    squares, counts, qualities = generate_random_squares(df, num_squares, us_shape)

    minerals = df['commodities'].unique()
    num_minerals = len(minerals)
    print(f"Minerals: {minerals}")
    print(f"Number of Minerals: {num_minerals}")
    
    with h5py.File(h5_file_path, 'w') as f:
        count_ds = f.create_dataset('counts', (num_squares, num_minerals, grid_size, grid_size), dtype='f')
        quality_ds = f.create_dataset('qualities', (num_squares, num_minerals, grid_size, grid_size), dtype='f')
        coord_ds = f.create_dataset('coordinates', (num_squares, 2), dtype='f')
        print(f"Shape of 'counts' dataset: {count_ds.shape}")
        print(f"Shape of 'qualities' dataset: {quality_ds.shape}")
        print(f"Shape of 'coordinates' dataset: {coord_ds.shape}")

        for idx, square in enumerate(squares):
            print(f"Processing square {idx + 1}/{num_squares}...")
            count_ds[idx] = counts[idx]
            quality_ds[idx] = qualities[idx]
            coord_ds[idx] = square
            
            # Save the model periodically
            if (idx + 1) % save_interval == 0:
                temp_path = f"{h5_file_path.rsplit('.', 1)[0]}_{idx + 1}.h5"
                print(f"Saving intermediate HDF5 file at {temp_path}...")
                with h5py.File(temp_path, 'w') as temp_f:
                    for name, ds in f.items():
                        temp_f.create_dataset(name, data=ds[:idx + 1])

    return squares

# Create the HDF5 file
squares = create_hdf5_file(h5_file_path, df, num_squares=100000)

print("Data preparation completed.")
# Visualization
def visualize_squares(squares, us_shape, output_path='generated_squares2.png'):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')
    
    for square in squares:
        lat_start, lon_start = square
        square_geom = box(lon_start, lat_start, lon_start + miles_to_degrees_lon(50, lat_start), lat_start + miles_to_degrees_lat(50))
        gpd.GeoSeries([square_geom]).boundary.plot(ax=ax, color='red')
    
    plt.title("Randomly Generated Squares Overlaying the US")
    plt.savefig(output_path)
    plt.close(fig)

# Example usage
# visualize_squares(squares, us_shape, output_path='/home/sujaynair/MRDS_Project/tilingVIS/generated_squares2.png')
