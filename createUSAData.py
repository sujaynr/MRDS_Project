import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import os
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from pyproj import CRS
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------- Configuration --------------------- #

# File paths for RaCA data
raca_file = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
output_raca_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/aggregated_racagrids.h5'  # Intermediate RaCA output

# File paths for mineral and geophysical data
mineral_csv_file = '/home/sujaynair/mrds.csv'
shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
fault_line_file = '/home/sujaynair/GDB/SHP/Qfaults_US_Database.shp'
geologic_units_path = '/home/sujaynair/GMNA_SHAPES/Geologic_units.shp'
elevation_data_dir = '/home/sujaynair/MRDS_Project/elevation_data'

data_dir = 'prepared_data_TILES'
combined_h5_file_path = os.path.join(data_dir, 'USADATA_combined.h5')  # Final combined output

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

# --------------------- Helper Functions --------------------- #

def miles_to_degrees_lat(miles):
    """Convert miles to degrees latitude."""
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    """Convert miles to degrees longitude based on latitude."""
    return miles / (69.0 * np.cos(np.radians(latitude)))

def convert_slip_rate(slip_rate):
    """Convert slip rate descriptions to numerical values."""
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

def get_geologic_info(lat, lon, gdf_usa, age_mapping, rocktype_mapping):
    """Retrieve geologic information for a given latitude and longitude."""
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

def get_elevation(lat, lon):
    """Placeholder function for elevation data retrieval."""
    # Replace with actual elevation data retrieval logic
    return 0

def check_fault_intersection(cell_bounds, fault_lines):
    """Check if a cell intersects with fault lines and compute slip rates."""
    faults_in_cell = fault_lines[fault_lines.intersects(cell_bounds)]
    if not faults_in_cell.empty:
        faults_presence = 1
        faults_slip_rate = faults_in_cell['slip_rate_numeric'].mean()
    else:
        faults_presence = 0
        faults_slip_rate = 0
    return faults_presence, faults_slip_rate

# --------------------- RaCA Model Definitions --------------------- #

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Define a 1D convolutional block."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

def conv2d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Define a 2D convolutional block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

class Scan1DModel(nn.Module):
    """1D Scan processing pipeline."""
    def __init__(self, num_wavelengths, num_lab_features):
        super(Scan1DModel, self).__init__()
        self.conv1 = conv1d_block(1, 32)
        self.conv2 = conv1d_block(32, 64)
        self.conv3 = conv1d_block(64, 64)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * (num_wavelengths // 2) + num_lab_features, 625)  # Adjusted input size
        self.output_size = (25, 25)  # Reshape target
    
    def forward(self, x_scan, x_lab):
        x = self.conv1(x_scan)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)  # [B,64,num_wavelengths//2]
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, x_lab], dim=1)  # Concatenate lab data
        x = self.fc(x)
        x = x.view(-1, self.output_size[0], self.output_size[1])  # [B,25,25]
        return x

class PedonConv2D(nn.Module):
    """Pedon-level 2D convolution."""
    def __init__(self):
        super(PedonConv2D, self).__init__()
        self.block = nn.Sequential(
            conv2d_block(1, 32),
            conv2d_block(32, 32),
            nn.AdaptiveAvgPool2d((25,25))
        )

    def forward(self, x):
        return self.block(x)

class SiteConv2D(nn.Module):
    """Site-level 2D convolution."""
    def __init__(self):
        super(SiteConv2D, self).__init__()
        self.block = nn.Sequential(
            conv2d_block(32,64),
            conv2d_block(64,64),
            nn.AdaptiveAvgPool2d((25,25))
        )

    def forward(self, x):
        return self.block(x)

class RegionConv2D(nn.Module):
    """Region-level 2D convolution."""
    def __init__(self):
        super(RegionConv2D, self).__init__()
        self.block = nn.Sequential(
            conv2d_block(64,64),
            conv2d_block(64,64),
            nn.AdaptiveAvgPool2d((25,25))
        )

    def forward(self, x):
        return self.block(x)

class FinalUpsampler(nn.Module):
    """Final upsampling and feature aggregation."""
    def __init__(self):
        super(FinalUpsampler, self).__init__()
        self.fc = nn.Linear(64 * 25 * 25, 288)
        self.upsample = nn.Upsample(size=(10,10), mode='bilinear', align_corners=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # to get [B,32]

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)  # [B,40000]
        x = self.fc(x)     # [B,288]
        x = x.view(B, 3, 3, 32)  # [B,3,3,32]
        x = x.permute(0,3,1,2)    # [B,32,3,3]
        x = self.upsample(x)      # [B,32,10,10]
        x = self.avgpool(x)       # [B,32,1,1]
        x = x.view(B,32)          # [B,32]
        return x

# --------------------- Load and Process RaCA Data --------------------- #

print("Loading RaCA data...")
full = pd.read_pickle(raca_file)

# Verify required columns
required_columns = ['rcasiteid_x', 'rcapid', 'sample_id', 'lat', 'long']
missing_columns = [col for col in required_columns if col not in full.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing in RaCA data: {missing_columns}")

# Rename columns for consistency
full['site_id'] = full['rcasiteid_x']  # Site ID
full['pedon_id'] = full['rcapid']      # Pedon ID
full['scan_id'] = full['sample_id']    # Scan ID

coordinate_columns = ['lat', 'long']
identifier_columns = ['site_id', 'pedon_id', 'scan_id']

wavelength_cols = [str(w) for w in range(365, 2501)]
lab_data_cols = ['adod', 'c_tot_ncs', 'caco3', 'n_tot_ncs', 's_tot_ncs', 'soc']

# Handle missing lab data columns
missing_lab_cols = [col for col in lab_data_cols if col not in full.columns]
if missing_lab_cols:
    print(f"The following lab data columns are missing in RaCA data and will be excluded: {missing_lab_cols}")
    lab_data_cols = [col for col in lab_data_cols if col in full.columns]

scan_lab_cols = wavelength_cols + lab_data_cols

print("Filtering RaCA data...")
full_filtered = full[coordinate_columns + identifier_columns + scan_lab_cols].copy()

print("Handling missing RaCA data...")
full_filtered.loc[:, scan_lab_cols] = full_filtered[scan_lab_cols].fillna(0)

print("Ensuring numeric coordinates in RaCA data...")
full_filtered['lat'] = pd.to_numeric(full_filtered['lat'], errors='coerce')
full_filtered['long'] = pd.to_numeric(full_filtered['long'], errors='coerce')

full_filtered = full_filtered.dropna(subset=['lat', 'long'])
print(f"RaCA Data after cleaning: {len(full_filtered)} records")
sites = full_filtered[['site_id', 'lat', 'long']].drop_duplicates()
print(f"Total number of unique RaCA sites: {len(sites)}")

# --------------------- Grid Generation --------------------- #

print("Loading shapefile for USA mainland...")
# Load the shapefile
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America'].geometry.unary_union

# Define the bounding box for the contiguous US mainland
contiguous_us_bbox = box(-125.0, 24.5, -66.0, 49.5)

# Intersect the US geometry with the bounding box to get the mainland shape
us_mainland_shape = us_shape.intersection(contiguous_us_bbox)

print("Generating grid of 50x50 mile squares covering the mainland US...")
# Approximate conversion from miles to degrees at the center latitude
avg_lat = (contiguous_us_bbox.bounds[1] + contiguous_us_bbox.bounds[3]) / 2
lat_step = miles_to_degrees_lat(50)
lon_step = miles_to_degrees_lon(50, avg_lat)

latitudes = np.arange(contiguous_us_bbox.bounds[1], contiguous_us_bbox.bounds[3], lat_step)
longitudes = np.arange(contiguous_us_bbox.bounds[0], contiguous_us_bbox.bounds[2], lon_step)

squares = []

for lat_start in latitudes:
    for lon_start in longitudes:
        # Define the square polygon
        lat_length = miles_to_degrees_lat(50)
        lon_length = miles_to_degrees_lon(50, lat_start)
        square_geom = box(lon_start, lat_start, lon_start + lon_length, lat_start + lat_length)
        
        # Check if the square intersects with the mainland US
        if us_mainland_shape.intersects(square_geom):
            squares.append((lat_start, lon_start))

num_squares = len(squares)
print(f"Total number of USA squares generated: {num_squares}")

# --------------------- Fault Lines and Geologic Units --------------------- #

print("Loading fault line data...")
fault_lines = gpd.read_file(fault_line_file)
fault_lines['slip_rate_numeric'] = fault_lines['slip_rate'].apply(convert_slip_rate)

print("Loading geologic units data...")
gdf = gpd.read_file(geologic_units_path)
# Define the custom CRS
custom_crs = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=-100 +k=0.926 +x_0=0 +y_0=0 +a=6371204 +b=6371204 +units=m +datum=WGS84")
gdf = gdf.set_crs(custom_crs, allow_override=True)
# Transform to EPSG:4326 (latitude and longitude)
gdf = gdf.to_crs('EPSG:4326')
# Crop the geologic units to the mainland US boundary
# Fix CRS mismatch warning by setting CRS for the right GeoDataFrame
mainland_gdf = gpd.GeoDataFrame(geometry=[us_mainland_shape], crs='EPSG:4326')
gdf_usa = gpd.overlay(gdf, mainland_gdf, how='intersection')

# Define the age mapping dictionary with approximate ages in millions of years ago (Ma)
age_mapping = {
    'Cambrian': 541,
    'Ordovician': 485,
    'Silurian': 443,
    'Devonian': 419,
    'Carboniferous': 359,
    'Permian': 299,
    'Triassic': 252,
    'Jurassic': 201,
    'Cretaceous': 145,
    'Paleogene': 66,
    'Neogene': 23,
    'Quaternary': 2.58
}

# Define the rock type mapping dictionary
rocktype_mapping = {
    'Sedimentary': 0,
    'Plutonic': 1,
    'Metamorphic and undivided crystalline': 2,
    'Volcanic': 3,
    'Ice': 4
}

# --------------------- Load and Process Mineral Data --------------------- #

print("Loading and processing the mineral dataset...")
# Load the mineral dataset
df = pd.read_csv(mineral_csv_file, low_memory=False)
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
num_minerals = len(specific_minerals)  # 10

# --------------------- RaCA Model Initialization --------------------- #

print("\nInitializing RaCA convolutional models...")
num_wavelengths = len(wavelength_cols)
num_lab_features = len(lab_data_cols)

# Initialize models
scan_model = Scan1DModel(num_wavelengths, num_lab_features).to(device)
pedon_model = PedonConv2D().to(device)
site_model = SiteConv2D().to(device)
region_model = RegionConv2D().to(device)
final_upsampler = FinalUpsampler().to(device)

# Set models to evaluation mode
scan_model.eval()
pedon_model.eval()
site_model.eval()
region_model.eval()
final_upsampler.eval()

# --------------------- Process RaCA Data --------------------- #

print("\nStarting RaCA data processing for all USA squares...")

# Initialize aggregated RaCA output
pixel_grid_size = 50 
final_feature_dim = 32  # after final pooling
aggregated_output = np.zeros((num_squares, pixel_grid_size, pixel_grid_size, final_feature_dim), dtype=np.float32)

for square_idx, square in enumerate(squares):
    lat_min, lon_min = square
    lat_max = lat_min + miles_to_degrees_lat(50)
    lon_max = lon_min + miles_to_degrees_lon(50, lat_min)

    print(f"\nProcessing RaCA square {square_idx + 1}/{num_squares} with bounds:")
    print(f"Latitude: {lat_min} to {lat_max}")
    print(f"Longitude: {lon_min} to {lon_max}")

    square_data = full_filtered[
        (full_filtered['lat'] >= lat_min) & (full_filtered['lat'] < lat_max) &
        (full_filtered['long'] >= lon_min) & (full_filtered['long'] < lon_max)
    ]

    print(f"Total number of RaCA records within square {square_idx + 1}: {len(square_data)}")

    if len(square_data) == 0:
        print(f"No RaCA data found within square {square_idx + 1}. Skipping.")
        continue

    lat_bins = np.linspace(lat_min, lat_max, pixel_grid_size + 1)
    lon_bins = np.linspace(lon_min, lon_max, pixel_grid_size + 1)

    for i in range(pixel_grid_size):
        for j in range(pixel_grid_size):
            lat_lower = lat_bins[i]
            lat_upper = lat_bins[i + 1]
            lon_lower = lon_bins[j]
            lon_upper = lon_bins[j + 1]

            pixel_data = square_data[
                (square_data['lat'] >= lat_lower) & (square_data['lat'] < lat_upper) &
                (square_data['long'] >= lon_lower) & (square_data['long'] < lon_upper)
            ]

            if len(pixel_data) == 0:
                continue

            pixel_sum = None
            sites_in_pixel = pixel_data['site_id'].unique()
            for site_id in sites_in_pixel:
                site_data = pixel_data[pixel_data['site_id'] == site_id]
                pedons_in_site = site_data['pedon_id'].unique()

                site_sum = None
                for pedon_id in pedons_in_site:
                    pedon_data = site_data[site_data['pedon_id'] == pedon_id]
                    scans_in_pedon = pedon_data['scan_id'].unique()

                    pedon_sum = None
                    for scan_id in scans_in_pedon:
                        scan_row = pedon_data[pedon_data['scan_id'] == scan_id]

                        scan_spectral = scan_row[wavelength_cols].values.astype(np.float32)
                        scan_lab = scan_row[lab_data_cols].values.astype(np.float32)

                        scan_spectral_tensor = torch.tensor(scan_spectral, device=device)
                        scan_lab_tensor = torch.tensor(scan_lab, device=device)

                        if scan_spectral_tensor.ndim == 1:
                            scan_spectral_tensor = scan_spectral_tensor.unsqueeze(0)
                        scan_spectral_tensor = scan_spectral_tensor.unsqueeze(1)  # [1,1,num_wavelengths]

                        if scan_lab_tensor.ndim == 1:
                            scan_lab_tensor = scan_lab_tensor.unsqueeze(0)  # [1,num_lab_features]

                        with torch.no_grad():
                            scan_out = scan_model(scan_spectral_tensor, scan_lab_tensor)  # [1,25,25]

                        if pedon_sum is None:
                            pedon_sum = scan_out
                        else:
                            pedon_sum += scan_out

                    if pedon_sum is not None:
                        pedon_sum = pedon_sum.unsqueeze(1)  # [1,1,25,25]
                        with torch.no_grad():
                            pedon_out = pedon_model(pedon_sum)  # [1,32,25,25]
                        if site_sum is None:
                            site_sum = pedon_out
                        else:
                            site_sum += pedon_out

                if site_sum is not None:
                    with torch.no_grad():
                        site_out = site_model(site_sum)  # [1,64,25,25]
                    if pixel_sum is None:
                        pixel_sum = site_out
                    else:
                        pixel_sum += site_out

            if pixel_sum is not None:
                with torch.no_grad():
                    region_out = region_model(pixel_sum)  # [1,64,25,25]
                    final_out = final_upsampler(region_out)  # [1,32]
                pixel_features = final_out.cpu().numpy()[0]  # [32]
                aggregated_output[square_idx, i, j, :] = pixel_features
                print(f"Processed RaCA pixel ({i+1}/{pixel_grid_size}, {j+1}/{pixel_grid_size}) in square {square_idx + 1}")
            # else: pixel remains zero vector

print("\nRaCA processing complete. Saving RaCA results...")
with h5py.File(output_raca_file, 'w') as f:
    f.create_dataset('aggregated_output', data=aggregated_output)
print(f"Aggregated RaCA output saved to: {output_raca_file}")
print(f"Aggregated RaCA Output Shape: {aggregated_output.shape}")
print(f"Aggregated RaCA Output Data Type: {aggregated_output.dtype}")

# --------------------- Combined Data Processing --------------------- #

# Define the total number of layers
additional_layers = 5  # Fault presence, slip rate, min_age, max_age, rock_type
raca_features = final_feature_dim  # 32
total_layers = num_minerals + additional_layers + raca_features  # 10 + 5 + 32 = 47

print(f"\nTotal Layers (10 minerals + 5 geophysical + 32 RaCA): {total_layers}")
print(f"Number of Squares: {num_squares}")

# Initialize combined_output with zeros
combined_output = np.zeros((num_squares, total_layers, pixel_grid_size, pixel_grid_size), dtype=np.float32)

# Function to fill mineral and geophysical layers
def fill_mineral_geophysical_layers(df, square, grid_size=50, cell_size=1, specific_minerals=specific_minerals, mineral_map=mineral_map, fault_lines=fault_lines, gdf_usa=gdf_usa, age_mapping=age_mapping, rocktype_mapping=rocktype_mapping):
    lat_start, lon_start = square
    num_minerals = len(specific_minerals)
    additional_layers = 5  # Fault presence, slip rate, min_age, max_age, rock_type

    # Initialize the data array with zeros
    data_array = np.zeros((num_minerals + additional_layers, grid_size, grid_size))

    # Precompute latitude boundaries for the cells
    lat_bins = np.linspace(lat_start, lat_start + miles_to_degrees_lat(50), grid_size + 1)
    
    # Compute longitude boundaries based on each cell's latitude
    lon_bins = np.zeros(grid_size + 1)
    for i in range(grid_size + 1):
        current_lat = lat_start + i * miles_to_degrees_lat(50) / grid_size
        lon_bins[i] = lon_start + i * miles_to_degrees_lon(50, current_lat) / grid_size

    # Pre-filter dataset for the square
    square_lat_min = lat_start
    square_lat_max = lat_start + miles_to_degrees_lat(50)
    square_lon_min = lon_start
    square_lon_max = lon_start + miles_to_degrees_lon(50, lat_start)
    square_data = df[
        (df['latitude'] >= square_lat_min) & (df['latitude'] < square_lat_max) &
        (df['longitude'] >= square_lon_min) & (df['longitude'] < square_lon_max)
    ]

    if not square_data.empty:
        # For each mineral, create a subset of data
        mineral_data_dict = {mineral: square_data[square_data['commodities'] == mineral] for mineral in specific_minerals}
    else:
        mineral_data_dict = {mineral: pd.DataFrame(columns=df.columns) for mineral in specific_minerals}

    # Process each cell
    for i in range(grid_size):
        lat_min_cell = lat_bins[i]
        lat_max_cell = lat_bins[i + 1]
        
        for j in range(grid_size):
            lon_min_cell = lon_bins[j]
            lon_max_cell = lon_bins[j + 1]
            
            # Cell boundaries
            cell_bounds = box(lon_min_cell, lat_min_cell, lon_max_cell, lat_max_cell)
            
            # Mineral counts
            for mineral in specific_minerals:
                mineral_idx = mineral_map[mineral]
                mineral_data = mineral_data_dict[mineral]
                cell_mineral_data = mineral_data[
                    (mineral_data['latitude'] >= lat_min_cell) & (mineral_data['latitude'] < lat_max_cell) &
                    (mineral_data['longitude'] >= lon_min_cell) & (mineral_data['longitude'] < lon_max_cell)
                ]
                count = len(cell_mineral_data)
                data_array[mineral_idx, i, j] = count  # Mineral layers at indices 0-9
            
            # Faults (presence and slip rate)
            faults_presence, faults_slip_rate = check_fault_intersection(cell_bounds, fault_lines)
            data_array[num_minerals, i, j] = faults_presence  # Fault presence at index 10
            data_array[num_minerals + 1, i, j] = faults_slip_rate  # Slip rate at index 11
            
            # Geologic Info
            center_lat = (lat_min_cell + lat_max_cell) / 2
            center_lon = (lon_min_cell + lon_max_cell) / 2
            min_age, max_age, rock_type = get_geologic_info(center_lat, center_lon, gdf_usa, age_mapping, rocktype_mapping)
            data_array[num_minerals + 2, i, j] = min_age      # Min age at index 12
            data_array[num_minerals + 3, i, j] = max_age      # Max age at index 13
            data_array[num_minerals + 4, i, j] = rock_type    # Rock type at index 14

    return data_array

# Function to combine mineral/geophysical and RaCA data
def process_combined_square(square_idx, square, df, aggregated_output, combined_output, grid_size=50, cell_size=1):
    if (square_idx + 1) % 10 == 0 or square_idx == 0:
        print(f"Processing combined square {square_idx + 1}/{num_squares}...")
    
    # Fill mineral and geophysical layers
    mineral_geophys_data = fill_mineral_geophysical_layers(
        df, square, grid_size, cell_size,
        specific_minerals=specific_minerals,
        mineral_map=mineral_map,
        fault_lines=fault_lines,
        gdf_usa=gdf_usa,
        age_mapping=age_mapping,
        rocktype_mapping=rocktype_mapping
    )
    
    # Retrieve RaCA data for this square
    racagrid = aggregated_output[square_idx, :, :, :]  # Shape: [50, 50, 32]
    
    # Transpose RaCA data to match (features, grid_x, grid_y)
    racagrid_transposed = racagrid.transpose(2, 0, 1)  # [32, 50, 50]
    
    # Concatenate along the feature axis
    combined_cell_data = np.concatenate((mineral_geophys_data, racagrid_transposed), axis=0)  # [47,50,50]
    
    # Assign to combined_output
    combined_output[square_idx] = combined_cell_data

    return

print("\nStarting combined data processing for all squares...")

# Initialize combined_output with zeros
# Initialize combined_output with zeros and ensure it's writable
combined_output = np.zeros((num_squares, total_layers, pixel_grid_size, pixel_grid_size), dtype=np.float32)
combined_output.setflags(write=True)

print("\nStarting combined data processing for all squares...")

# Process squares sequentially (ignoring parallelization)
for square_idx, square in enumerate(squares):
    if (square_idx + 1) % 10 == 0 or square_idx == 0:
        print(f"Processing combined square {square_idx + 1}/{num_squares}...")
    process_combined_square(
        square_idx=square_idx,
        square=square,
        df=df,
        aggregated_output=aggregated_output,
        combined_output=combined_output,
        grid_size=pixel_grid_size,
        cell_size=1
    )

print("\nCombined data processing complete. Saving combined results...")

# Save the combined data to HDF5
with h5py.File(combined_h5_file_path, 'w') as f:
    f.create_dataset('data', data=combined_output)
    f.create_dataset('USA_coords', data=np.array(squares))
print(f"Combined data saved to: {combined_h5_file_path}")
print(f"Combined Data Shape: {combined_output.shape}")
print(f"Combined Data Type: {combined_output.dtype}")


# --------------------- Visualization (Optional) --------------------- #

def visualize_combined_square(combined_output, square_index, output_path='combined_square.png'):
    """Visualize the combined features of a single square."""
    square_data = combined_output[square_index]  # Shape: [47, 50, 50]
    
    # Separate features
    mineral_geophys = square_data[:15, :, :]  # [15,50,50]
    raca_features = square_data[15:, :, :]    # [32,50,50]
    
    # Example visualization: Sum of mineral/geophysical features and sum of RaCA features
    mineral_sum = np.sum(mineral_geophys, axis=0)
    raca_sum = np.sum(raca_features, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(mineral_sum, cmap='viridis', interpolation='nearest')
    plt.title(f'Square {square_index} - Sum of Mineral/Geophysical Features')
    plt.colorbar(label='Summed Values')
    plt.xlabel('Longitude Pixel')
    plt.ylabel('Latitude Pixel')
    
    plt.subplot(1, 2, 2)
    plt.imshow(raca_sum, cmap='plasma', interpolation='nearest')
    plt.title(f'Square {square_index} - Sum of RaCA Features')
    plt.colorbar(label='Summed Values')
    plt.xlabel('Longitude Pixel')
    plt.ylabel('Latitude Pixel')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage (Optional)
# visualize_combined_square(combined_output, 0, output_path='/home/sujaynair/MRDS_Project/tilingVIS/combined_square_0.png')
