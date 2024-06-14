# import wandb
# import os
# import pickle
# import pdb
# import numpy as np
# import torch
# from scipy.ndimage import gaussian_filter
# import geopandas as gpd
# from shapely.geometry import box
# from utils import gaussian_smooth_and_normalize, visualize_layers, compute_dice_coefficients, plot_dice_coefficients, plot_metric
# from models import MineralTransformer

# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # Initialize wandb
# wandb.init(project="mineral_transformer_project", name="2048_0.0001TEST")

# grid_size = 60
# d_model = 512
# x_min, x_max = -125, -66.5
# y_min, y_max = 24.5, 49.5
# pixel_size_x = (x_max - x_min) / grid_size
# pixel_size_y = (y_max - y_min) / grid_size

# shapefile_path = '/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
# countries = gpd.read_file(shapefile_path)

# usa = countries[countries['ADMIN'] == 'United States of America']

# grid_cells = []
# for i in range(grid_size):
#     for j in range(grid_size):
#         lon_min = x_min + j * pixel_size_x
#         lon_max = lon_min + pixel_size_x
#         lat_max = y_max - i * pixel_size_y
#         lat_min = lat_max - pixel_size_y
#         cell = box(lon_min, lat_min, lon_max, lat_max)
#         grid_cells.append(cell)

# grid = gpd.GeoDataFrame({'geometry': grid_cells})
# grid.crs = usa.crs

# ### MASKING
# mask_us = np.zeros((grid_size, grid_size), dtype=bool)
# mask_non_us_land = np.zeros((grid_size, grid_size), dtype=bool)
# mask_ocean = np.zeros((grid_size, grid_size), dtype=bool)

# for i in range(grid_size):
#     for j in range(grid_size):
#         cell = grid.geometry[i * grid_size + j]
#         if any(usa.intersects(cell)):
#             mask_us[i, j] = True
#         elif any(countries.intersects(cell)):
#             mask_non_us_land[i, j] = True
#         else:
#             mask_ocean[i, j] = True
# mask_layer = np.zeros((grid_size, grid_size), dtype=int)
# mask_layer[mask_us] = 1
# mask_layer[mask_non_us_land] = 2

# fig, ax = plt.subplots(figsize=(10, 8))
# usa.boundary.plot(ax=ax, linewidth=1)
# grid.boundary.plot(ax=ax, color='gray', linestyle='--', linewidth=0.5)
# # Plot masks
# gdf_mask_us = gpd.GeoDataFrame({'geometry': grid.geometry[mask_us.flatten()]})
# gdf_mask_us.boundary.plot(ax=ax, color='blue', linewidth=1, label='US')

# gdf_mask_non_us_land = gpd.GeoDataFrame({'geometry': grid.geometry[mask_non_us_land.flatten()]})
# gdf_mask_non_us_land.boundary.plot(ax=ax, color='green', linewidth=1, label='Non-US Land')

# gdf_mask_ocean = gpd.GeoDataFrame({'geometry': grid.geometry[mask_ocean.flatten()]})
# gdf_mask_ocean.boundary.plot(ax=ax, color='red', linewidth=1, label='Ocean')

# plt.legend()
# plt.title('US, Non-US Land, and Ocean Masks')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()
# ###

# def weighted_mse_loss(pred, target, weight):
#     return torch.mean(weight * (pred - target) ** 2)

# losses = []
# def train(model, input_tensor_train, output_tensor_train, mask_layer, num_epochs=100, learning_rate=0.0001):
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#     criterion = weighted_mse_loss

#     weight_mask = np.ones(mask_layer.shape)
#     weight_mask[mask_layer == 1] = 1.0
#     weight_mask[mask_layer != 1] = 0.0
#     weight_mask_tensor = torch.tensor(weight_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(input_tensor_train, output_tensor_train)

#         loss = criterion(outputs.view(-1, grid_size * grid_size) * weight_mask_tensor.view(-1, grid_size * grid_size),
#                          output_tensor_train.view(-1, grid_size * grid_size) * weight_mask_tensor.view(-1, grid_size * grid_size),
#                          weight_mask_tensor.view(-1, grid_size * grid_size))
        
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         losses.append(loss.item())
#         wandb.log({"loss": loss.item()})

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

#     plt.plot(losses, label='Loss')
#     plt.title('Training Loss Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('trainingVis/loss_curve.png')
#     plt.close()

#     wandb.save('trainingVis/loss_curve.png')

# data_dir = 'prepared_dataClipped60'
# elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
# data = {}

# for elem in elements:
#     with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
#         data[elem] = pickle.load(f)

# for elem in elements:
#     print(f"{elem} - A layer sum: {np.sum(data[elem][0])}")
#     print(f"{elem} - B layer sum: {np.sum(data[elem][1])}")
#     print(f"{elem} - C layer sum: {np.sum(data[elem][2])}")
#     print(f"{elem} - D layer sum: {np.sum(data[elem][3])}")
#     print(f"{elem} - E layer sum: {np.sum(data[elem][4])}")

# input_elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper']
# output_elements = ['Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

# input_layers = np.stack([data[elem] for elem in input_elements], axis=0)
# output_layers = np.stack([data[elem] for elem in output_elements], axis=0)

# # Size of input_layers: (5, 5, 60, 60)
# # Size of output_layers: (5, 5, 60, 60)

# mask_layer_expanded = np.expand_dims(mask_layer, axis=0)  # Add a new axis at the start
# # Size of mask_layer_expanded: (1, 60, 60)
# mask_layer_expanded = np.expand_dims(mask_layer_expanded, axis=0)  # Add another axis for batch
# # Size of mask_layer_expanded: (1, 1, 60, 60)
# mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[0], axis=0)  # Repeat for batch size
# # Size of mask_layer_expanded: (5, 1, 60, 60)
# mask_layer_expanded = np.repeat(mask_layer_expanded, input_layers.shape[1], axis=1)  # Repeat for sequence length
# # Size of mask_layer_expanded: (5, 5, 60, 60)

# input_layers_with_mask = np.concatenate((input_layers, mask_layer_expanded), axis=1)
# # Size of input_layers_with_mask: (5, 10, 60, 60)

# def process_layers(layers, smooth=False):
#     if smooth:
#         return np.array([gaussian_smooth_and_normalize(layer) for layer in layers])
#     else:
#         return layers

# # Flag to determine whether to smooth the data
# smooth_data = True

# smoothed_input_layers = process_layers(input_layers_with_mask, smooth=smooth_data)
# smoothed_output_layers = process_layers(output_layers, smooth=smooth_data)
# # Size of smoothed_input_layers: (5, 10, 60, 60)
# # Size of smoothed_output_layers: (5, 5, 60, 60)

# batch_size = smoothed_input_layers.shape[0] # 5
# sequence_length_input = smoothed_input_layers.shape[1] # 10
# sequence_length_output = smoothed_output_layers.shape[1] # 5
# feature_dimension = grid_size * grid_size # 3600

# smoothed_input_layers = smoothed_input_layers.reshape(batch_size, sequence_length_input, feature_dimension)
# smoothed_output_layers = smoothed_output_layers.reshape(batch_size, sequence_length_output, feature_dimension)
# # Size of smoothed_input_layers: (5, 10, 3600)
# # Size of smoothed_output_layers: (5, 5, 3600)

# input_tensor_train = torch.tensor(smoothed_input_layers, dtype=torch.float32)
# output_tensor_train = torch.tensor(smoothed_output_layers, dtype=torch.float32)
# # Size of input_tensor_train: torch.Size([5, 10, 3600])
# # Size of output_tensor_train: torch.Size([5, 5, 3600])


# # Visualize layers
# def visualize_layer(layer_data, title):
#     fig, axs = plt.subplots(1, 5, figsize=(20, 4))
#     for i, ax in enumerate(axs):
#         ax.imshow(layer_data[i], cmap='hot', interpolation='nearest')
#         ax.set_title(f'{title} Layer {chr(65+i)}')
#     plt.show()

# # Visualize input elements
# for elem in input_elements:
#     visualize_layer(data[elem], elem)

# # Visualize output elements
# for elem in output_elements:
#     visualize_layer(data[elem], elem)

# model = MineralTransformer(d_model=d_model)
# train(model, input_tensor_train, output_tensor_train, mask_layer)

# model.eval()
# with torch.no_grad():
#     predicted_output_train = model(input_tensor_train, output_tensor_train)

# input_np_train = input_tensor_train.numpy().reshape(batch_size, sequence_length_input, grid_size, grid_size)
# output_np_train = output_tensor_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)
# predicted_np_train = predicted_output_train.numpy().reshape(batch_size, sequence_length_output, grid_size, grid_size)
# smoothed_predicted_np_train = np.array([gaussian_smooth_and_normalize(predicted_np_train[i]) for i in range(predicted_np_train.shape[0])])

# us_mask = mask_us.flatten()

# flattened_output_np_train = output_np_train.reshape(batch_size, sequence_length_output, -1)
# flattened_predicted_np_train = smoothed_predicted_np_train.reshape(batch_size, sequence_length_output, -1)

# output_np_train_us = flattened_output_np_train[:, :, us_mask]
# predicted_np_train_us = flattened_predicted_np_train[:, :, us_mask]

# # Logging metric values to wandb
# metric_values = []
# for i in range(sequence_length_output):
#     predicted_sum = np.sum(predicted_np_train_us[0][i])
#     ground_truth_sum = np.sum(output_np_train_us[0][i])
#     metric = predicted_sum / ground_truth_sum
#     metric_values.append(metric)
#     wandb.log({f'Layer_{chr(65+i)}_Metric': metric})
#     print(f'Layer {chr(65+i)}: Metric = {metric}')

# plot_metric(metric_values, [chr(65+i) for i in range(sequence_length_output)])
# wandb.save('metric_plot.png')  # Save the plot to wandb

# # VIS
# for i in range(5):  # By quality
#     visualize_layers(i, input_np_train, output_np_train, smoothed_predicted_np_train, input_elements, output_elements)

# # Dice coefficients for each layer in the US
# dice_coeffs = compute_dice_coefficients(predicted_np_train_us, output_np_train_us, threshold=0.05)
# plot_dice_coefficients(dice_coeffs, output_elements)
# wandb.save('dice_coefficients_plot.png')  # Save the plot to wandb

# # Finish the wandb run
# wandb.finish()
import wandb
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

# Initialize wandb
wandb.init(project="mineral_transformer_project", name="2048_0.0001TEST")

grid_size = 60
d_model = 1024
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

def weighted_mse_loss(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)

losses = []
def train(model, input_tensor_train, output_tensor_train, mask_layer, num_epochs=200, learning_rate=0.0001):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = weighted_mse_loss

    weight_mask = np.ones(mask_layer.shape)
    weight_mask[mask_layer == 1] = 1.0
    weight_mask[mask_layer == 2] = 0.1
    weight_mask[mask_layer == 0] = 0.0
    weight_mask_tensor = torch.tensor(weight_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    batch_size, seq_length_input, feature_dim = input_tensor_train.shape
    seq_length_output = output_tensor_train.shape[1]

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor_train, output_tensor_train)

        # Apply mask to outputs to ensure no predictions in non-USA regions
        mask_tensor = torch.tensor(mask_us, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(outputs.device)
        masked_outputs = outputs * mask_tensor

        # Normalize outputs to match the target total counts by layer
        normalized_outputs = []
        for i in range(seq_length_output):
            output_sum_layer = masked_outputs[:, i, :, :].sum(dim=(1, 2), keepdim=True)
            target_sum_layer = output_tensor_train[:, i, :].sum(dim=1, keepdim=True).unsqueeze(-1)
            normalization_factor = target_sum_layer / output_sum_layer
            normalized_outputs.append(masked_outputs[:, i, :, :] * normalization_factor)

        normalized_outputs = torch.stack(normalized_outputs, dim=1)

        # Flatten for loss computation
        flat_normalized_outputs = normalized_outputs.view(-1, grid_size * grid_size)
        flat_output_tensor_train = output_tensor_train.view(-1, grid_size * grid_size)

        loss = criterion(flat_normalized_outputs, flat_output_tensor_train, weight_mask_tensor.view(-1, grid_size * grid_size))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        wandb.log({"loss": loss.item()})

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    plt.plot(losses, label='Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('trainingVis/loss_curve.png')
    plt.close()

    wandb.save('trainingVis/loss_curve.png')


data_dir = 'prepared_dataClipped60'
elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)

for elem in elements:
    print(f"{elem} - A layer sum: {np.sum(data[elem][0])}")
    print(f"{elem} - B layer sum: {np.sum(data[elem][1])}")
    print(f"{elem} - C layer sum: {np.sum(data[elem][2])}")
    print(f"{elem} - D layer sum: {np.sum(data[elem][3])}")
    print(f"{elem} - E layer sum: {np.sum(data[elem][4])}")

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

def process_layers(layers, smooth=False):
    if smooth:
        return np.array([gaussian_smooth_and_normalize(layer) for layer in layers])
    else:
        return layers

# Flag to determine whether to smooth the data
smooth_data = True

smoothed_input_layers = process_layers(input_layers_with_mask, smooth=smooth_data)
smoothed_output_layers = process_layers(output_layers, smooth=smooth_data)
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

# Visualize layers
def visualize_layer(layer_data, title):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axs):
        ax.imshow(layer_data[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'{title} Layer {chr(65+i)}')
    plt.show()

# Visualize input elements
# for elem in input_elements:
#     visualize_layer(data[elem], elem)

# # Visualize output elements
# for elem in output_elements:
#     visualize_layer(data[elem], elem)

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

# Logging metric values to wandb
metric_values = []
for i in range(sequence_length_output):
    predicted_sum = np.sum(predicted_np_train_us[0][i])
    ground_truth_sum = np.sum(output_np_train_us[0][i])
    metric = predicted_sum / ground_truth_sum
    metric_values.append(metric)
    wandb.log({f'Layer_{chr(65+i)}_Metric': metric})
    print(f'Layer {chr(65+i)}: Metric = {metric}')

plot_metric(metric_values, [chr(65+i) for i in range(sequence_length_output)])
wandb.save('metric_plot.png')  # Save the plot to wandb

# VIS
for i in range(5):  # By quality
    visualize_layers(i, input_np_train, output_np_train, smoothed_predicted_np_train, input_elements, output_elements)

# Dice coefficients for each layer in the US
dice_coeffs = compute_dice_coefficients(predicted_np_train_us, output_np_train_us, threshold=0.05)
plot_dice_coefficients(dice_coeffs, output_elements)
wandb.save('dice_coefficients_plot.png')  # Save the plot to wandb

def visualize_predictions(predictions, ground_truth, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(predictions.reshape(grid_size, grid_size), cmap='hot', interpolation='nearest')
    axs[0].set_title(f'{title} Predictions')
    axs[1].imshow(ground_truth.reshape(grid_size, grid_size), cmap='hot', interpolation='nearest')
    axs[1].set_title(f'{title} Ground Truth')
    plt.show()

# for i in range(sequence_length_output):
#     visualize_predictions(predicted_np_train_us[0][i], output_np_train_us[0][i], f'Layer {chr(65+i)}')

# Finish the wandb run
wandb.finish()
