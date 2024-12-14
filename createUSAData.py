import pandas as pd
import numpy as np
import h5py
import os
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from pyproj import CRS

# File paths
file_path = '/home/sujaynair/mrds.csv'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
fault_line_file = '/home/sujaynair/GDB/SHP/Qfaults_US_Database.shp'
geologic_units_path = '/home/sujaynair/GMNA_SHAPES/Geologic_units.shp'
elevation_data_dir = '/home/sujaynair/MRDS_Project/elevation_data'

data_dir = 'prepared_data_TILES'
h5_file_path = os.path.join(data_dir, 'USADATA.h5')

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

print("Loading shapefile...")
# Load the shapefile
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America'].geometry.unary_union

# Define the bounding box for the contiguous US mainland
contiguous_us_bbox = box(-125.0, 24.5, -66.0, 49.5)

# Intersect the US geometry with the bounding box to get the mainland shape
us_mainland_shape = us_shape.intersection(contiguous_us_bbox)

# Convert miles to degrees latitude and longitude
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

print("Loading and processing the mineral dataset...")
# Load the mineral dataset
df = pd.read_csv(file_path, low_memory=False)
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

# Map minerals to indices
mineral_map = {mineral: idx for idx, mineral in enumerate(specific_minerals)}

# Get the bounding box of the mainland US from the bounding box we defined
min_lon, min_lat, max_lon, max_lat = contiguous_us_bbox.bounds

# Adjust the bounding box slightly to ensure coverage
min_lat -= 1
max_lat += 1
min_lon -= 1
max_lon += 1

# Generate a grid of squares covering the mainland US
print("Generating grid of squares covering the mainland US...")
# Approximate conversion from miles to degrees at the center latitude
avg_lat = (min_lat + max_lat) / 2
lat_step = miles_to_degrees_lat(50)
lon_step = miles_to_degrees_lon(50, avg_lat)

latitudes = np.arange(min_lat, max_lat, lat_step)
longitudes = np.arange(min_lon, max_lon, lon_step)

squares = []
USA_coords = []

for lat_start in latitudes:
    for lon_start in longitudes:
        # Define the square polygon
        lat_length = miles_to_degrees_lat(50)
        lon_length = miles_to_degrees_lon(50, lat_start)
        square_geom = box(lon_start, lat_start, lon_start + lon_length, lat_start + lat_length)
        
        # Check if the square intersects with the mainland US
        if us_mainland_shape.intersects(square_geom):
            squares.append((lat_start, lon_start))
            USA_coords.append((lat_start, lon_start))

print(f"Total number of squares: {len(squares)}")

# Precompute latitude lengths and longitude lengths for cell_size=1 mile
cell_size = 1  # 1 mile cells
grid_size = 50  # 50x50 grid
lat_cell_size = miles_to_degrees_lat(cell_size)

# Load Fault Lines
print("Loading fault line data...")
fault_lines = gpd.read_file(fault_line_file)

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

# Load Geologic Units
print("Loading geologic units data...")
gdf = gpd.read_file(geologic_units_path)
# Define the custom CRS
custom_crs = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=-100 +k=0.926 +x_0=0 +y_0=0 +a=6371204 +b=6371204 +units=m +datum=WGS84")
gdf = gdf.set_crs(custom_crs, allow_override=True)
# Transform to EPSG:4326 (latitude and longitude)
gdf = gdf.to_crs('EPSG:4326')
# Crop the geologic units to the mainland US boundary
gdf_usa = gpd.overlay(gdf, gpd.GeoDataFrame(geometry=[us_mainland_shape]), how='intersection')

# Define the age mapping dictionary with approximate ages in millions of years ago (Ma)
age_mapping = {
    # Include your complete age mapping dictionary here
    # For brevity, the dictionary content is omitted.
}

# Define the rock type mapping dictionary
rocktype_mapping = {
    'Sedimentary': 0,
    'Plutonic': 1,
    'Metamorphic and undivided crystalline': 2,
    'Volcanic': 3,
    'Ice': 4
}

# Function to get geologic unit information for a given lat/lon
def get_geologic_info(lat, lon):
    point = Point(lon, lat)
    result = gdf_usa[gdf_usa.geometry.contains(point)]
    if not result.empty:
        min_age_str = result.iloc[0]['MIN_AGE']
        max_age_str = result.iloc[0]['MAX_AGE']
        rocktype_str = result.iloc[0]['ROCKTYPE']
        min_age = age_mapping.get(min_age_str, 0)
        max_age = age_mapping.get(max_age_str, 0)
        rocktype = rocktype_mapping.get(rocktype_str, 0)
        return min_age, max_age, rocktype
    else:
        return 0, 0, 0

# Elevation Data Handling (Placeholder Function)
def get_elevation(lat, lon):
    # Placeholder: Replace with actual elevation data retrieval
    # For now, return 0 as elevation
    return 0

# Check if cell intersects with fault lines and get slip rates
def check_fault_intersection(cell_bounds):
    faults_in_cell = fault_lines[fault_lines.intersects(cell_bounds)]
    if not faults_in_cell.empty:
        faults_presence = 1
        # For fault layer, we'll use the presence (1) as one layer
        faults_slip_rate = faults_in_cell['slip_rate_numeric'].mean()
    else:
        faults_presence = 0
        faults_slip_rate = 0
    return faults_presence, faults_slip_rate

# Optimized fill_cells function
def fill_cells_optimized(df, square, grid_size=50, cell_size=1):
    lat_start, lon_start = square
    minerals = specific_minerals
    num_minerals = len(minerals)
    total_layers = num_minerals + 5  # 10 minerals + 5 additional layers

    # Initialize the combined data array with shape (15, grid_size, grid_size)
    data_array = np.zeros((total_layers, grid_size, grid_size))

    # Precompute latitude boundaries for the cells
    lat_indices = np.arange(grid_size + 1)
    latitudes = lat_start + lat_indices * lat_cell_size

    # Precompute longitude degrees per mile for each latitude
    lon_cell_sizes = miles_to_degrees_lon(cell_size, latitudes)

    # Precompute longitude boundaries for the cells
    lon_indices = np.zeros((grid_size + 1, grid_size + 1))
    for i in range(grid_size + 1):
        lon_lengths = lon_cell_sizes[i]
        lon_indices[i, :] = lon_start + np.cumsum(np.insert(lon_lengths * np.ones(grid_size), 0, 0))

    # Pre-filter dataset for the square
    square_lat_min = lat_start
    square_lat_max = latitudes[-1]
    square_lon_min = lon_start
    square_lon_max = lon_indices[-1, -1]
    square_data = df[
        (df['latitude'] >= square_lat_min) & (df['latitude'] < square_lat_max) &
        (df['longitude'] >= square_lon_min) & (df['longitude'] < square_lon_max)
    ]

    if not square_data.empty:
        # For each mineral, create a subset of data
        mineral_data_dict = {mineral: square_data[square_data['commodities'] == mineral] for mineral in minerals}
    else:
        mineral_data_dict = {mineral: pd.DataFrame(columns=df.columns) for mineral in minerals}

    # Process each cell
    for i in range(grid_size):
        lat_min = latitudes[i]
        lat_max = latitudes[i + 1]
        
        for j in range(grid_size):
            lon_min = lon_indices[i, j]
            lon_max = lon_indices[i, j + 1]
            
            # Cell boundaries
            cell_bounds = box(lon_min, lat_min, lon_max, lat_max)
            
            # Mineral counts
            for mineral in minerals:
                mineral_idx = mineral_map[mineral]
                mineral_data = mineral_data_dict[mineral]
                cell_mineral_data = mineral_data[
                    (mineral_data['latitude'] >= lat_min) & (mineral_data['latitude'] < lat_max) &
                    (mineral_data['longitude'] >= lon_min) & (mineral_data['longitude'] < lon_max)
                ]
                count = len(cell_mineral_data)
                data_array[mineral_idx, i, j] = count  # Mineral layers at indices 0-9
            
            # Faults (presence layer)
            faults_presence, faults_slip_rate = check_fault_intersection(cell_bounds)
            data_array[10, i, j] = faults_presence  # Fault presence at index 10
            
            # Geologic Info
            min_age, max_age, rock_type = get_geologic_info((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
            data_array[11, i, j] = min_age      # Min age at index 11
            data_array[12, i, j] = max_age      # Max age at index 12
            data_array[13, i, j] = rock_type    # Rock type at index 13
            
            # Elevation (Placeholder)
            data_array[14, i, j] = get_elevation((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)  # Elevation at index 14

    return data_array

# Move process_square to global scope
def process_square(idx_square, df, grid_size, cell_size, num_squares):
    idx, square = idx_square
    if idx % 10 == 0:
        print(f"Processing square {idx + 1}/{num_squares}...")
    data_array = fill_cells_optimized(df, square, grid_size, cell_size)
    return idx, data_array, square

# Main function to create the HDF5 file
def create_hdf5_file(h5_file_path, df, squares, grid_size=50, cell_size=1):
    print("Processing squares and creating HDF5 file...")
    
    minerals = specific_minerals
    num_minerals = len(minerals)
    total_layers = num_minerals + 5  # 15 layers total
    num_squares = len(squares)
    print(f"Minerals: {minerals}")
    print(f"Number of Minerals: {num_minerals}")
    print(f"Total Layers (including additional features): {total_layers}")
    print(f"Number of Squares: {num_squares}")
    
    with h5py.File(h5_file_path, 'w') as f:
        data_ds = f.create_dataset('data', (num_squares, total_layers, grid_size, grid_size), dtype='f')
        coord_ds = f.create_dataset('USA_coords', (num_squares, 2), dtype='f')
        print(f"Shape of 'data' dataset: {data_ds.shape}")
        print(f"Shape of 'USA_coords' dataset: {coord_ds.shape}")
    
        # Prepare list of indices and squares
        idx_squares = list(enumerate(squares))
        
        # Process squares in parallel
        process_square_partial = partial(
            process_square, df=df, grid_size=grid_size, cell_size=cell_size, num_squares=num_squares
        )
        
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(process_square_partial)(idx_square) for idx_square in idx_squares
        )
        
        # Save results to datasets
        for idx, data_array, square in results:
            data_ds[idx] = data_array
            coord_ds[idx] = square
    
    return

# Create the HDF5 file
create_hdf5_file(h5_file_path, df, squares)

print("Data preparation completed.")

# Visualization (Optional)
def visualize_squares(squares, us_mainland_shape, output_path='usa_mainland_tiling.png'):
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot mainland US boundary
    gpd.GeoSeries([us_mainland_shape]).boundary.plot(ax=ax, color='black')
    
    # For better performance, create a GeoSeries of all squares at once
    square_geoms = []
    for square in squares:
        lat_start, lon_start = square
        lat_length = miles_to_degrees_lat(50)
        lon_length = miles_to_degrees_lon(50, lat_start)
        square_geom = box(lon_start, lat_start, lon_start + lon_length, lat_start + lat_length)
        square_geoms.append(square_geom)
    
    # Create a GeoDataFrame for all squares
    squares_gdf = gpd.GeoDataFrame(geometry=square_geoms)
    squares_gdf.boundary.plot(ax=ax, color='red', linewidth=0.5)
    
    plt.title("50x50 Mile Squares Covering the Mainland US")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

# Example usage (Optional)
# visualize_squares(squares, us_mainland_shape, output_path='/home/sujaynair/MRDS_Project/tilingVIS/usa_mainland_tiling.png')
