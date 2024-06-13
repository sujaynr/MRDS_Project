import os
import pickle
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import box
from utils import gaussian_smooth_and_normalize, visualize_layers, compute_dice_coefficients, plot_dice_coefficients, plot_metric
from models import MineralTransformer

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


grid_size = 60
d_model = 512
x_min, x_max = -125, -66.5
y_min, y_max = 24.5, 49.5
pixel_size_x = (x_max - x_min) / grid_size
pixel_size_y = (y_max - y_min) / grid_size


shapefile_path = '/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
countries = gpd.read_file(shapefile_path)

usa = countries[countries['ADMIN'] == 'United States of America']

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

### MASKING
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
mask_layer = np.zeros((grid_size, grid_size), dtype=int)
mask_layer[mask_us] = 1
mask_layer[mask_non_us_land] = 2

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
###

losses = []
def train(model, input_tensor_train, output_tensor_train, mask_layer, num_epochs=500, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Weight mask for the loss
    weight_mask = np.ones(mask_layer.shape)
    weight_mask[mask_layer == 1] = 1.0
    weight_mask[mask_layer != 1] = 0.0  # Non-US land and ocean regions
    weight_mask_tensor = torch.tensor(weight_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence length dimensions

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor_train, output_tensor_train)

        # Apply the weight mask to the loss
        loss = criterion(outputs.view(-1, grid_size * grid_size) * weight_mask_tensor.view(-1, grid_size * grid_size),
                         output_tensor_train.view(-1, grid_size * grid_size) * weight_mask_tensor.view(-1, grid_size * grid_size))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        losses.append(loss.item())  # Append the loss value to the curve

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Plot the loss curve
    plt.plot(losses, label='Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('trainingVis/loss_curve.png')
    plt.close()


data_dir = 'prepared_dataClipped60'
elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)

input_elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper']
output_elements = ['Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

input_layers = np.stack([data[elem] for elem in input_elements], axis=0)
output_layers = np.stack([data[elem] for elem in output_elements], axis=0)

# Size of input_layers: (5, 5, 60, 60)
# Size of output_layers: (5, 5, 60, 60)

mask_layer_expanded = np.expand_dims(mask_layer, axis=0)  # Add a new axis at the start
# Size of mask_layer_expanded: (1, 60, 60)
mask_layer_expanded = np.expand_dims(mask_layer_expanded, axis=0)  # Add another axis for batch
# Size of mask_layer_expanded: (1, 1, 60, 60)
mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[0], axis=0)  # Repeat for batch size
# Size of mask_layer_expanded: (5, 1, 60, 60)
mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[1], axis=1)  # Repeat for sequence length
# Size of mask_layer_expanded: (5, 5, 60, 60)

input_layers_with_mask = np.concatenate((input_layers, mask_layer_expanded), axis=1)
# Size of input_layers_with_mask: (5, 10, 60, 60)

smoothed_input_layers = np.array([gaussian_smooth_and_normalize(layer) for layer in input_layers_with_mask])
smoothed_output_layers = np.array([gaussian_smooth_and_normalize(layer) for layer in output_layers])
# Size of smoothed_input_layers: (5, 10, 60, 60)
# Size of smoothed_output_layers: (5, 5, 60, 60)

batch_size = smoothed_input_layers.shape[0] # 5
sequence_length_input = smoothed_input_layers.shape[1] # 10
sequence_length_output = smoothed_output_layers.shape[1] # 5
feature_dimension = grid_size * grid_size # 3600

smoothed_input_layers = smoothed_input_layers.reshape(batch_size, sequence_length_input, feature_dimension)
smoothed_output_layers = smoothed_output_layers.reshape(batch_size, sequence_length_output, feature_dimension)
# Size of smoothed_input_layers: (5, 10, 3600)
# Size of smoothed_output_layers: (5, 5, 3600)


input_tensor_train = torch.tensor(smoothed_input_layers, dtype=torch.float32)
output_tensor_train = torch.tensor(smoothed_output_layers, dtype=torch.float32)
# Size of input_tensor_train: torch.Size([5, 10, 3600])
# Size of output_tensor_train: torch.Size([5, 5, 3600])

model = MineralTransformer(d_model=d_model)
train(model, input_tensor_train, output_tensor_train, mask_layer)

model.eval()
with torch.no_grad():
    predicted_output_train = model(input_tensor_train, output_tensor_train)

input_np_train = input_tensor_train.numpy().reshape(batch_size, sequence_length_input, grid_size, grid_size)
output_np_train = output_tensor_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)
predicted_np_train = predicted_output_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)
smoothed_predicted_np_train = np.array([gaussian_smooth_and_normalize(predicted_np_train[i]) for i in range(predicted_np_train.shape[0])])


us_mask = mask_us.flatten()

flattened_output_np_train = output_np_train.reshape(batch_size, sequence_length_output, -1)
flattened_predicted_np_train = smoothed_predicted_np_train.reshape(batch_size, sequence_length_output, -1)

output_np_train_us = flattened_output_np_train[:, :, us_mask]
predicted_np_train_us = flattened_predicted_np_train[:, :, us_mask]


metric_values = []
for i in range(sequence_length_output):
    predicted_sum = np.sum(predicted_np_train_us[0][i])
    ground_truth_sum = np.sum(output_np_train_us[0][i])
    metric = predicted_sum / ground_truth_sum
    metric_values.append(metric)
    print(f'Layer {chr(65+i)}: Metric = {metric}')

plot_metric(metric_values, [chr(65+i) for i in range(sequence_length_output)])

# VIS
for i in range(5):  # By quality
    visualize_layers(i, input_np_train, output_np_train, smoothed_predicted_np_train, input_elements, output_elements)

# Dice coefficients for each layer in the US
dice_coeffs = compute_dice_coefficients(predicted_np_train_us, output_np_train_us, threshold=0.05)
plot_dice_coefficients(dice_coeffs, output_elements)