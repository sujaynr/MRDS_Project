# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import h5py
# import os
# import matplotlib.pyplot as plt

# file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
# mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
# output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagridsFILLED.h5'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print("Loading RaCA data...")
# full = pd.read_pickle(file1)

# required_columns = ['rcasiteid_x', 'rcapid', 'sample_id', 'lat', 'long']
# missing_columns = [col for col in required_columns if col not in full.columns]
# if missing_columns:
#     raise ValueError(f"The following required columns are missing: {missing_columns}")

# full['site_id'] = full['rcasiteid_x']  # Site ID
# full['pedon_id'] = full['rcapid']      # Pedon ID
# full['scan_id'] = full['sample_id']    # Scan ID

# coordinate_columns = ['lat', 'long']
# identifier_columns = ['site_id', 'pedon_id', 'scan_id']

# wavelength_cols = [str(w) for w in range(365, 2501)]
# lab_data_cols = ['adod', 'c_tot_ncs', 'caco3', 'n_tot_ncs', 's_tot_ncs', 'soc']

# missing_lab_cols = [col for col in lab_data_cols if col not in full.columns]
# if missing_lab_cols:
#     print(f"The following lab data columns are missing and will be excluded: {missing_lab_cols}")
#     lab_data_cols = [col for col in lab_data_cols if col in full.columns]

# scan_lab_cols = wavelength_cols + lab_data_cols

# print("Filtering data...")
# full_filtered = full[coordinate_columns + identifier_columns + scan_lab_cols].copy()

# print("Handling missing data...")
# full_filtered.loc[:, scan_lab_cols] = full_filtered[scan_lab_cols].fillna(0)

# print("Ensuring numeric coordinates...")
# full_filtered.loc[:, 'lat'] = pd.to_numeric(full_filtered['lat'], errors='coerce')
# full_filtered.loc[:, 'long'] = pd.to_numeric(full_filtered['long'], errors='coerce')

# full_filtered = full_filtered.dropna(subset=['lat', 'long'])
# print(f"Data after cleaning: {len(full_filtered)} records")
# sites = full_filtered[['site_id', 'lat', 'long']].drop_duplicates()
# print(f"Total number of unique sites: {len(sites)}")
# print("Loading coordinates of squares...")
# with h5py.File(mineral_file, 'r') as f:
#     coords = f['coordinates'][:]


# coords_lat = coords[:, 0]
# coords_lon = coords[:, 1]

# print("\nCoordinate Ranges:")
# print(f"Coords latitude range: {coords_lat.min()} to {coords_lat.max()}")
# print(f"Coords longitude range: {coords_lon.min()} to {coords_lon.max()}")
# print(f"RaCA data latitude range: {sites['lat'].min()} to {sites['lat'].max()}")
# print(f"RaCA data longitude range: {sites['long'].min()} to {sites['long'].max()}")

# miles_per_degree = 69.0 
# degree_offset = 50 / miles_per_degree 

# squares_with_data_indices = []

# print("\nIdentifying squares containing RaCA data...")
# for idx in range(len(coords)):
#     lat_min = coords[idx, 0]
#     lon_min = coords[idx, 1]
#     lat_max = lat_min + degree_offset
#     lon_max = lon_min + degree_offset
#     sites_in_square = sites[
#         (sites['lat'] >= lat_min) & (sites['lat'] < lat_max) &
#         (sites['long'] >= lon_min) & (sites['long'] < lon_max)
#     ]

#     if len(sites_in_square) > 0:
#         squares_with_data_indices.append(idx)

# print(f"Total number of squares: {len(coords)}")
# print(f"Number of squares containing RaCA data: {len(squares_with_data_indices)}")

# # Define convolutional models
# class Scan1DConv(nn.Module):
#     def __init__(self, num_wavelengths, num_lab_features, output_size=(8, 8)):
#         super(Scan1DConv, self).__init__()
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         conv_output_length = num_wavelengths // 2
#         self.fc = nn.Linear(64 * conv_output_length + num_lab_features, output_size[0] * output_size[1])
#         self.output_size = output_size

#     def forward(self, x_scan, x_lab):
#         x = self.conv1d(x_scan)
#         x = x.view(x.size(0), -1)
#         x = torch.cat([x, x_lab], dim=1)
#         x = self.fc(x)
#         x = x.view(-1, self.output_size[0], self.output_size[1])
#         return x

# class Conv2DReduction(nn.Module):
#     def __init__(self, in_channels, out_channels, output_size=(8, 8)):
#         super(Conv2DReduction, self).__init__()
#         self.conv2d = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=output_size)
#         )

#     def forward(self, x):
#         x = self.conv2d(x)
#         return x

# num_wavelengths = len(wavelength_cols)
# num_lab_features = len(lab_data_cols)
# output_size = (8, 8)

# print("\nInitializing convolutional models...")
# scan_conv = Scan1DConv(num_wavelengths, num_lab_features, output_size=output_size).to(device)
# pedon_conv = Conv2DReduction(in_channels=1, out_channels=32, output_size=output_size).to(device)
# site_conv = Conv2DReduction(in_channels=32, out_channels=64, output_size=output_size).to(device)
# pixel_conv = Conv2DReduction(in_channels=64, out_channels=64, output_size=(1, 1)).to(device) 

# scan_conv.eval()
# pedon_conv.eval()
# site_conv.eval()
# pixel_conv.eval()

# total_squares = len(coords)
# pixel_grid_size = 50 
# final_feature_dim = 64 

# # (10000, 50, 50, 64)
# aggregated_output = np.zeros((total_squares, pixel_grid_size, pixel_grid_size, final_feature_dim), dtype=np.float32)

# print("\nStarting processing of all squares...")

# for square_idx in squares_with_data_indices:
#     lat_min = coords_lat[square_idx]
#     lon_min = coords_lon[square_idx]
#     lat_max = lat_min + degree_offset
#     lon_max = lon_min + degree_offset

#     processed_square_idx = squares_with_data_indices.index(square_idx)

#     print(f"\nProcessing square {square_idx} with bounds:")
#     print(f"Latitude: {lat_min} to {lat_max}")
#     print(f"Longitude: {lon_min} to {lon_max}")

#     square_data = full_filtered[
#         (full_filtered['lat'] >= lat_min) & (full_filtered['lat'] < lat_max) &
#         (full_filtered['long'] >= lon_min) & (full_filtered['long'] < lon_max)
#     ]

#     print(f"Total number of records within square {square_idx}: {len(square_data)}")

#     if len(square_data) == 0:
#         print(f"No data found within square {square_idx}. Skipping.")
#         continue

#     lat_bins = np.linspace(lat_min, lat_max, pixel_grid_size + 1)
#     lon_bins = np.linspace(lon_min, lon_max, pixel_grid_size + 1)

#     for i in range(pixel_grid_size):
#         for j in range(pixel_grid_size):
#             lat_lower = lat_bins[i]
#             lat_upper = lat_bins[i + 1]
#             lon_lower = lon_bins[j]
#             lon_upper = lon_bins[j + 1]

#             pixel_data = square_data[
#                 (square_data['lat'] >= lat_lower) & (square_data['lat'] < lat_upper) &
#                 (square_data['long'] >= lon_lower) & (square_data['long'] < lon_upper)
#             ]

#             if len(pixel_data) == 0:
#                 continue

#             pixel_sum = None

#             # Process each site within the pixel
#             sites_in_pixel = pixel_data['site_id'].unique()
#             for site_id in sites_in_pixel:
#                 site_data = pixel_data[pixel_data['site_id'] == site_id]
#                 pedons_in_site = site_data['pedon_id'].unique()

#                 site_sum = None

#                 # Process each pedon within the site
#                 for pedon_id in pedons_in_site:
#                     pedon_data = site_data[site_data['pedon_id'] == pedon_id]
#                     scans_in_pedon = pedon_data['scan_id'].unique()

#                     pedon_sum = None

#                     # Process each scan within the pedon
#                     for scan_id in scans_in_pedon:
#                         scan_row = pedon_data[pedon_data['scan_id'] == scan_id]

#                         scan_spectral = scan_row[wavelength_cols].values.astype(np.float32)
#                         scan_lab = scan_row[lab_data_cols].values.astype(np.float32)

#                         scan_spectral_tensor = torch.tensor(scan_spectral).to(device)
#                         scan_lab_tensor = torch.tensor(scan_lab).to(device)

#                         # check tensorr shape
#                         if scan_spectral_tensor.ndim == 1:
#                             scan_spectral_tensor = scan_spectral_tensor.unsqueeze(0)
#                         scan_spectral_tensor = scan_spectral_tensor.unsqueeze(1)  # [1, 1, num_wavelengths]

#                         if scan_lab_tensor.ndim == 1:
#                             scan_lab_tensor = scan_lab_tensor.unsqueeze(0)  # 1, num_lab_features]

#                         # 1D convolution
#                         with torch.no_grad():
#                             scan_output = scan_conv(scan_spectral_tensor, scan_lab_tensor)  # [1, 8, 8]

#                         if pedon_sum is None:
#                             pedon_sum = scan_output
#                         else:
#                             pedon_sum += scan_output

#                     if pedon_sum is not None:
#                         # Add channel dimension for Conv2D
#                         pedon_sum = pedon_sum.unsqueeze(1)  #[1, 1, 8, 8]
#                         with torch.no_grad():
#                             pedon_output = pedon_conv(pedon_sum)  # [1, 32, 8, 8]
#                         if site_sum is None:
#                             site_sum = pedon_output
#                         else:
#                             site_sum += pedon_output

#             if site_sum is not None:
#                 with torch.no_grad():
#                     site_output = site_conv(site_sum)  # [1, 64, 8, 8]
#                 if pixel_sum is None:
#                     pixel_sum = site_output
#                 else:
#                     pixel_sum += site_output

#             if pixel_sum is not None:
#                 with torch.no_grad():
#                     pixel_output = pixel_conv(pixel_sum)  # [1, 64, 1, 1]
#                 pixel_features = pixel_output.view(pixel_output.size(0), -1).cpu().numpy()  # [1, 64]
#                 aggregated_output[square_idx, i, j, :] = pixel_features[0]
#                 print(f"Processed pixel ({i+1}/{pixel_grid_size}, {j+1}/{pixel_grid_size}) in square {square_idx}")
#             else:
#                 # Pixel remains as zero vector
#                 pass

#     print("\nProcessing complete. Saving results...")

#     with h5py.File(output_file, 'w') as f:
#         f.create_dataset('aggregated_output', data=aggregated_output)

#     print(f"Aggregated output saved to: {output_file}")
#     print(f"Aggregated Output Shape: {aggregated_output.shape}")
#     print(f"Aggregated Output Data Type: {aggregated_output.dtype}")


import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import os
import matplotlib.pyplot as plt

file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file_new = '/home/sujaynair/MRDS_Project/newRaCA_DATA/raca_raw.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading RaCA data...")
full = pd.read_pickle(file1)

required_columns = ['rcasiteid_x', 'rcapid', 'sample_id', 'lat', 'long']
missing_columns = [col for col in required_columns if col not in full.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")

full['site_id'] = full['rcasiteid_x']  # Site ID
full['pedon_id'] = full['rcapid']      # Pedon ID
full['scan_id'] = full['sample_id']    # Scan ID

coordinate_columns = ['lat', 'long']
identifier_columns = ['site_id', 'pedon_id', 'scan_id']

wavelength_cols = [str(w) for w in range(365, 2501)]
lab_data_cols = ['adod', 'c_tot_ncs', 'caco3', 'n_tot_ncs', 's_tot_ncs', 'soc']

missing_lab_cols = [col for col in lab_data_cols if col not in full.columns]
if missing_lab_cols:
    print(f"The following lab data columns are missing and will be excluded: {missing_lab_cols}")
    lab_data_cols = [col for col in lab_data_cols if col in full.columns]

scan_lab_cols = wavelength_cols + lab_data_cols

print("Filtering data...")
full_filtered = full[coordinate_columns + identifier_columns + scan_lab_cols].copy()

print("Handling missing data...")
full_filtered.loc[:, scan_lab_cols] = full_filtered[scan_lab_cols].fillna(0)

print("Ensuring numeric coordinates...")
full_filtered.loc[:, 'lat'] = pd.to_numeric(full_filtered['lat'], errors='coerce')
full_filtered.loc[:, 'long'] = pd.to_numeric(full_filtered['long'], errors='coerce')

full_filtered = full_filtered.dropna(subset=['lat', 'long'])
print(f"Data after cleaning: {len(full_filtered)} records")
sites = full_filtered[['site_id', 'lat', 'long']].drop_duplicates()
print(f"Total number of unique sites: {len(sites)}")
print("Loading coordinates of squares...")
with h5py.File(mineral_file, 'r') as f:
    coords = f['coordinates'][:]

coords_lat = coords[:, 0]
coords_lon = coords[:, 1]

print("\nCoordinate Ranges:")
print(f"Coords latitude range: {coords_lat.min()} to {coords_lat.max()}")
print(f"Coords longitude range: {coords_lon.min()} to {coords_lon.max()}")
print(f"RaCA data latitude range: {sites['lat'].min()} to {sites['lat'].max()}")
print(f"RaCA data longitude range: {sites['long'].min()} to {sites['long'].max()}")

def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))



miles_per_degree = 69.0
degree_offset = 50 / miles_per_degree

squares_with_data_indices = []

print("\nIdentifying squares containing RaCA data...")

output_raca = []

for idx in range(len(coords)):
    lat_start = coords[idx, 0]
    lon_start = coords[idx, 1]

    lat_data = []
    nonempty_cells = 0

    for i in range(50):
        lon_data = []
        for j in range(50):
            lat_min = lat_start + i * miles_to_degrees_lat(1)
            lat_max = lat_start + (i + 1) * miles_to_degrees_lat(1)
            lon_min = lon_start + j * miles_to_degrees_lon(1, lat_start)
            lon_max = lon_start + (j + 1) * miles_to_degrees_lon(1, lat_start)

            sites_in_square = sites[
                (sites['lat'] >= lat_min) & (sites['lat'] < lat_max) &
                (sites['long'] >= lon_min) & (sites['long'] < lon_max)
            ]
            full_data_rows = full_filtered[full_filtered['site_id'].isin(sites_in_square['site_id'])]
            raw_scans = []
            for index, row in full_data_rows.iterrows():
                scan_spectral = row[wavelength_cols].values.astype(np.float32)
                raw_scans.append(scan_spectral)
            if len(raw_scans) > 0:
                nonempty_cells += 1
            lon_data.append(raw_scans)
        lat_data.append(lon_data)
    print(f"Number of nonempty cells: {nonempty_cells} out of 2500")
    print(f"finished square {idx} out of 10000")
    output_raca.append(lat_data)
    # if len(output_raca) > 10:
    #     break
with open(output_file_new, "wb") as file:
    pickle.dump(output_raca, file)

# print(f"Total number of squares: {len(coords)}")
# print(f"Number of squares containing RaCA data: {len(squares_with_data_indices)}")

# #### Model Definitions ####

# def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#     return nn.Sequential(
#         nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.BatchNorm1d(out_channels),
#         nn.LeakyReLU(inplace=True)
#     )

# def conv2d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.BatchNorm2d(out_channels),
#         nn.LeakyReLU(inplace=True)
#     )

# class Scan1DModel(nn.Module):
#     """
#     1D Scan processing pipeline:
#     Input: [B,1,2136] scans, plus [B,6] lab data
#     - 1D conv layers + BN + LeakyReLU
#     - AvgPool1D(k=2,s=2)
#     Flatten + concat lab data = [B,68358]
#     Fully connected to [B,625]
#     Reshape [B,25,25]
#     """
#     def __init__(self, num_wavelengths, num_lab_features):
#         super(Scan1DModel, self).__init__()
#         self.conv1 = conv1d_block(1, 32)
#         self.conv2 = conv1d_block(32, 64)
#         self.conv3 = conv1d_block(64, 64)
#         self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(68352 + num_lab_features, 625)  # 68352 derived from (64*1068)

#     def forward(self, x_scan, x_lab):
#         x = self.conv1(x_scan)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.avgpool(x)  # [B,64,1068]

#         x = x.view(x.size(0), -1)  # [B,68352]
#         x = torch.cat([x, x_lab], dim=1)  # [B,68358]
#         x = self.fc(x)  # [B,625]
#         x = x.view(-1, 25, 25)  # [B,25,25]
#         return x

# class PedonConv2D(nn.Module):
#     """
#     Pedon-level:
#     Input: [B,1,25,25] after summing scans
#     2D conv+BN+LeakyReLU blocks + AdaptiveAvgPool2d((25,25))
#     Output: [B,32,25,25]
#     """
#     def __init__(self):
#         super(PedonConv2D, self).__init__()
#         self.block = nn.Sequential(
#             conv2d_block(1, 32),
#             conv2d_block(32, 32),
#             nn.AdaptiveAvgPool2d((25,25))
#         )

#     def forward(self, x):
#         return self.block(x)

# class SiteConv2D(nn.Module):
#     """
#     Site-level:
#     Input: [B,32,25,25] after summing pedons
#     Output: [B,64,25,25]
#     """
#     def __init__(self):
#         super(SiteConv2D, self).__init__()
#         self.block = nn.Sequential(
#             conv2d_block(32,64),
#             conv2d_block(64,64),
#             nn.AdaptiveAvgPool2d((25,25))
#         )

#     def forward(self, x):
#         return self.block(x)

# class RegionConv2D(nn.Module):
#     """
#     Region-level:
#     Input: [B,64,25,25] after summing sites
#     Output: [B,64,25,25] (final representation before flatten)
#     """
#     def __init__(self):
#         super(RegionConv2D, self).__init__()
#         self.block = nn.Sequential(
#             conv2d_block(64,64),
#             conv2d_block(64,64),
#             nn.AdaptiveAvgPool2d((25,25))
#         )

#     def forward(self, x):
#         return self.block(x)

# class FinalUpsampler(nn.Module):
#     """
#     After RegionConv2D: [B,64,25,25]
#     Flatten to [B,40000]
#     FC to [B,288]
#     Reshape to [B,3,3,32]
#     Upsample to [B,32,10,10]
#     Finally, we can average pool to get [B,32] for storage
#     """
#     def __init__(self):
#         super(FinalUpsampler, self).__init__()
#         self.fc = nn.Linear(40000, 288)
#         self.upsample = nn.Upsample(size=(10,10), mode='bilinear', align_corners=False)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # to get [B,32]

#     def forward(self, x):
#         # x: [B,64,25,25]
#         B = x.size(0)
#         x = x.view(B, -1)  # [B,40000]
#         x = self.fc(x)     # [B,288]
        
#         # Reshape [B,288]->[B,3,3,32]
#         x = x.view(B, 3, 3, 32)
#         # Permute to [B,32,3,3] for upsampling
#         x = x.permute(0,3,1,2) # [B,32,3,3]

#         # Upsample to [B,32,10,10]
#         x = self.upsample(x)

#         # Average pool spatial to get [B,32]
#         x = self.avgpool(x) # [B,32,1,1]
#         x = x.view(B,32)
#         return x

# num_wavelengths = len(wavelength_cols)
# num_lab_features = len(lab_data_cols)

# print("\nInitializing convolutional models...")
# scan_model = Scan1DModel(num_wavelengths, num_lab_features).to(device)
# pedon_model = PedonConv2D().to(device)
# site_model = SiteConv2D().to(device)
# region_model = RegionConv2D().to(device)
# final_upsampler = FinalUpsampler().to(device)

# scan_model.eval()
# pedon_model.eval()
# site_model.eval()
# region_model.eval()
# final_upsampler.eval()

# total_squares = len(coords)
# pixel_grid_size = 50 
# final_feature_dim = 32  # after final pooling

# # (10000, 50, 50, 32)
# aggregated_output = np.zeros((total_squares, pixel_grid_size, pixel_grid_size, final_feature_dim), dtype=np.float32)

# print("\nStarting processing of all squares...")

# for square_idx in squares_with_data_indices:
#     lat_min = coords_lat[square_idx]
#     lon_min = coords_lon[square_idx]
#     lat_max = lat_min + degree_offset
#     lon_max = lon_min + degree_offset

#     print(f"\nProcessing square {square_idx} with bounds:")
#     print(f"Latitude: {lat_min} to {lat_max}")
#     print(f"Longitude: {lon_min} to {lon_max}")

#     square_data = full_filtered[
#         (full_filtered['lat'] >= lat_min) & (full_filtered['lat'] < lat_max) &
#         (full_filtered['long'] >= lon_min) & (full_filtered['long'] < lon_max)
#     ]

#     print(f"Total number of records within square {square_idx}: {len(square_data)}")

#     if len(square_data) == 0:
#         print(f"No data found within square {square_idx}. Skipping.")
#         continue

#     lat_bins = np.linspace(lat_min, lat_max, pixel_grid_size + 1)
#     lon_bins = np.linspace(lon_min, lon_max, pixel_grid_size + 1)

#     for i in range(pixel_grid_size):
#         for j in range(pixel_grid_size):
#             lat_lower = lat_bins[i]
#             lat_upper = lat_bins[i + 1]
#             lon_lower = lon_bins[j]
#             lon_upper = lon_bins[j + 1]

#             pixel_data = square_data[
#                 (square_data['lat'] >= lat_lower) & (square_data['lat'] < lat_upper) &
#                 (square_data['long'] >= lon_lower) & (square_data['long'] < lon_upper)
#             ]

#             if len(pixel_data) == 0:
#                 continue

#             # Aggregate over sites
#             pixel_sum = None
#             sites_in_pixel = pixel_data['site_id'].unique()
#             for site_id in sites_in_pixel:
#                 site_data = pixel_data[pixel_data['site_id'] == site_id]
#                 pedons_in_site = site_data['pedon_id'].unique()

#                 site_sum = None
#                 for pedon_id in pedons_in_site:
#                     pedon_data = site_data[site_data['pedon_id'] == pedon_id]
#                     scans_in_pedon = pedon_data['scan_id'].unique()

#                     pedon_sum = None
#                     for scan_id in scans_in_pedon:
#                         scan_row = pedon_data[pedon_data['scan_id'] == scan_id]

#                         scan_spectral = scan_row[wavelength_cols].values.astype(np.float32)
#                         scan_lab = scan_row[lab_data_cols].values.astype(np.float32)

#                         scan_spectral_tensor = torch.tensor(scan_spectral, device=device)
#                         scan_lab_tensor = torch.tensor(scan_lab, device=device)

#                         if scan_spectral_tensor.ndim == 1:
#                             scan_spectral_tensor = scan_spectral_tensor.unsqueeze(0)
#                         scan_spectral_tensor = scan_spectral_tensor.unsqueeze(1)  # [1,1,num_wavs]

#                         if scan_lab_tensor.ndim == 1:
#                             scan_lab_tensor = scan_lab_tensor.unsqueeze(0)  # [1,num_lab_features]

#                         with torch.no_grad():
#                             scan_out = scan_model(scan_spectral_tensor, scan_lab_tensor)  # [1,25,25]

#                         # Sum scans within a pedon
#                         if pedon_sum is None:
#                             pedon_sum = scan_out
#                         else:
#                             pedon_sum += scan_out

#                     # Pedon-level conv
#                     if pedon_sum is not None:
#                         pedon_sum = pedon_sum.unsqueeze(1) # [1,1,25,25]
#                         with torch.no_grad():
#                             pedon_out = pedon_model(pedon_sum) # [1,32,25,25]

#                         # Sum pedons within a site
#                         if site_sum is None:
#                             site_sum = pedon_out
#                         else:
#                             site_sum += pedon_out

#                 # Site-level conv
#                 if site_sum is not None:
#                     with torch.no_grad():
#                         site_out = site_model(site_sum) # [1,64,25,25]
#                     # Sum sites within a pixel
#                     if pixel_sum is None:
#                         pixel_sum = site_out
#                     else:
#                         pixel_sum += site_out

#             # Region-level conv
#             if pixel_sum is not None:
#                 with torch.no_grad():
#                     region_out = region_model(pixel_sum) # [1,64,25,25]
#                     final_out = final_upsampler(region_out) # [1,32]

#                 pixel_features = final_out.cpu().numpy()[0] # [32]
#                 aggregated_output[square_idx, i, j, :] = pixel_features
#                 print(f"Processed pixel ({i+1}/{pixel_grid_size}, {j+1}/{pixel_grid_size}) in square {square_idx}")
#             # else: pixel stays zero

#     print("\nProcessing complete. Saving results...")

#     with h5py.File(output_file, 'w') as f:
#         f.create_dataset('aggregated_output', data=aggregated_output)

#     print(f"Aggregated output saved to: {output_file}")
#     print(f"Aggregated Output Shape: {aggregated_output.shape}")
#     print(f"Aggregated Output Data Type: {aggregated_output.dtype}")


## TESTING:


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
