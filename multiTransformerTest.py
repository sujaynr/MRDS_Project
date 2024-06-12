import os
import pickle
import numpy as np
import torch
import pdb
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import box

from utils import gaussian_smooth_and_normalize, visualize_layers, compute_dice_coefficients, plot_dice_coefficients
from models import MineralTransformer

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


grid_size = 30
d_model = 512

# Define bounds and grid size
grid_size = 30
x_min, x_max = -125, -66.5
y_min, y_max = 24.5, 49.5
pixel_size_x = (x_max - x_min) / grid_size
pixel_size_y = (y_max - y_min) / grid_size

# Read shapefile
shapefile_path = '/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
countries = gpd.read_file(shapefile_path)

# Filter for the United States
usa = countries[countries['ADMIN'] == 'United States of America']

# Create the grid cells
grid_cells = []
for i in range(grid_size):
    for j in range(grid_size):
        lon_min = x_min + j * pixel_size_x
        lon_max = lon_min + pixel_size_x
        lat_max = y_max - i * pixel_size_y
        lat_min = lat_max - pixel_size_y
        cell = box(lon_min, lat_min, lon_max, lat_max)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame({'geometry': grid_cells})
grid.crs = usa.crs

# Create masks for US, non-US land, and ocean
mask_us = np.zeros((grid_size, grid_size), dtype=bool)
mask_non_us_land = np.zeros((grid_size, grid_size), dtype=bool)
mask_ocean = np.zeros((grid_size, grid_size), dtype=bool)

for i in range(grid_size):
    for j in range(grid_size):
        cell = grid.geometry[i * grid_size + j]
        if any(usa.intersects(cell)):
            mask_us[i, j] = True
        elif any(countries.intersects(cell)):
            mask_non_us_land[i, j] = True
        else:
            mask_ocean[i, j] = True

# Create a combined mask layer: 0 for ocean, 1 for US, 2 for non-US land
mask_layer = np.zeros((grid_size, grid_size), dtype=int)
mask_layer[mask_us] = 1
mask_layer[mask_non_us_land] = 2

# Visualize the masked area
fig, ax = plt.subplots(figsize=(10, 8))
usa.boundary.plot(ax=ax, linewidth=1)
grid.boundary.plot(ax=ax, color='gray', linestyle='--', linewidth=0.5)

# Plot masks
gdf_mask_us = gpd.GeoDataFrame({'geometry': grid.geometry[mask_us.flatten()]})
gdf_mask_us.boundary.plot(ax=ax, color='blue', linewidth=1, label='US')

gdf_mask_non_us_land = gpd.GeoDataFrame({'geometry': grid.geometry[mask_non_us_land.flatten()]})
gdf_mask_non_us_land.boundary.plot(ax=ax, color='green', linewidth=1, label='Non-US Land')

gdf_mask_ocean = gpd.GeoDataFrame({'geometry': grid.geometry[mask_ocean.flatten()]})
gdf_mask_ocean.boundary.plot(ax=ax, color='red', linewidth=1, label='Ocean')

plt.legend()
plt.title('US, Non-US Land, and Ocean Masks')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


def train(model, input_tensor_train, output_tensor_train, num_epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor_train, output_tensor_train)
        loss = criterion(outputs.view(-1, grid_size * grid_size), output_tensor_train.view(-1, grid_size * grid_size))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# Load data and integrate the mask layer
data_dir = 'prepared_data'
elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Clay', 'Copper', 'Sand and Gravel', 'Stone', 'Vanadium']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)

input_elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper']
output_elements = ['Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

input_layers = np.stack([data[elem] for elem in input_elements], axis=0)
output_layers = np.stack([data[elem] for elem in output_elements], axis=0)


mask_layer_expanded = np.expand_dims(mask_layer, axis=0)  # Add a new axis at the start
mask_layer_expanded = np.expand_dims(mask_layer_expanded, axis=0)  # Add another axis for batch
mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[0], axis=0)  # Repeat for batch size
mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[1], axis=1)  # Repeat for sequence length


input_layers_with_mask = np.concatenate((input_layers, mask_layer_expanded), axis=1)
pdb.set_trace()

smoothed_input_layers = np.array([gaussian_smooth_and_normalize(layer) for layer in input_layers_with_mask])
smoothed_output_layers = np.array([gaussian_smooth_and_normalize(layer) for layer in output_layers])


print("smoothed_input_layers shape:", smoothed_input_layers.shape)
print("smoothed_output_layers shape:", smoothed_output_layers.shape)


batch_size = smoothed_input_layers.shape[0]
sequence_length_input = smoothed_input_layers.shape[1]
sequence_length_output = smoothed_output_layers.shape[1]
feature_dimension = grid_size * grid_size

smoothed_input_layers = smoothed_input_layers.reshape(batch_size, sequence_length_input, feature_dimension)
smoothed_output_layers = smoothed_output_layers.reshape(batch_size, sequence_length_output, feature_dimension)


print("Reshaped smoothed input layers shape:", smoothed_input_layers.shape)
print("Reshaped smoothed output layers shape:", smoothed_output_layers.shape)

# Convert to tensors
input_tensor_train = torch.tensor(smoothed_input_layers, dtype=torch.float32)
output_tensor_train = torch.tensor(smoothed_output_layers, dtype=torch.float32)

print("Train input tensor shape:", input_tensor_train.shape)
print("Train output tensor shape:", output_tensor_train.shape)


model = MineralTransformer(d_model=d_model)


train(model, input_tensor_train, output_tensor_train)

model.eval()
with torch.no_grad():
    predicted_output_train = model(input_tensor_train, output_tensor_train)

input_np_train = input_tensor_train.numpy().reshape(batch_size, sequence_length_input, grid_size, grid_size)
output_np_train = output_tensor_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)
predicted_np_train = predicted_output_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)

# Apply Gaussian smoothing and normalization to the predicted output
smoothed_predicted_np_train = np.array([gaussian_smooth_and_normalize(predicted_np_train[i]) for i in range(predicted_np_train.shape[0])])

# Visualization and evaluation functions


for i in range(5): # By quality
    visualize_layers(i, input_np_train, output_np_train, smoothed_predicted_np_train, input_elements, output_elements)

for i in range(5):
    predicted_sum = np.sum(predicted_np_train[0][i])
    ground_truth_sum = np.sum(output_np_train[0][i])
    metric = predicted_sum / ground_truth_sum
    print(f'Layer {chr(65+i)}: Metric = {metric}')


# Dice coefficients for each layer
dice_coeffs = compute_dice_coefficients(smoothed_predicted_np_train, output_np_train, threshold=0.05)

for layer_index, dice_per_layer in dice_coeffs.items():
    print(f'Quality Layer {chr(65+layer_index)}: Dice Coefficients = {dice_per_layer}')



plot_dice_coefficients(dice_coeffs, output_elements)
