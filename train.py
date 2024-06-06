import os
import pickle
import pdb
import numpy as np
import cv2
import torch
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from models import MineralTransformer

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set the conda environment
conda_env = '/Users/sujaynair/anaconda3/envs/dataAnalysis'
os.environ['CONDA_PREFIX'] = conda_env


grid_size = 30
d_model = 512

def gaussian_smooth_and_normalize(layers, sigma=1.0):
    smoothed_layers = []
    for layer in layers:
        smoothed_layer = gaussian_filter(layer, sigma=sigma)
        normalized_layer = smoothed_layer * (layer.sum() / smoothed_layer.sum())
        smoothed_layers.append(normalized_layer)
    return np.stack(smoothed_layers, axis=0)

data_dir = 'prepared_data'
elements = ['Gold', 'Silver']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)


input_elements = ['Gold']
output_elements = ['Silver']

input_layers = np.stack([data[elem] for elem in input_elements], axis=0)
output_layers = np.stack([data[elem] for elem in output_elements], axis=0)

# Normalize the data
input_layers = (input_layers - input_layers.mean()) / input_layers.std()
output_layers = (output_layers - output_layers.mean()) / output_layers.std()

print("Initial input layers shape:", input_layers.shape)
print("Initial output layers shape:", output_layers.shape)

# Reshape the input and output to be 3D: (batch_size, sequence_length, feature_dimension)
batch_size = input_layers.shape[0] # todo, THIS SHOULD BE ABLE TO TAKE ANY VALUE
sequence_length = input_layers.shape[1]
feature_dimension = grid_size * grid_size

input_layers = input_layers.reshape(batch_size, sequence_length, feature_dimension)
output_layers = output_layers.reshape(batch_size, sequence_length, feature_dimension)

print("Reshaped input layers shape:", input_layers.shape)
print("Reshaped output layers shape:", output_layers.shape)

input_tensor = torch.tensor(input_layers, dtype=torch.float32)
output_tensor = torch.tensor(output_layers, dtype=torch.float32).view(batch_size, sequence_length, grid_size, grid_size)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)

model = MineralTransformer(d_model=d_model)

def train(model, input_tensor, output_tensor, num_epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor, output_tensor)
        loss = criterion(outputs, output_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(input_tensor, output_tensor)
            val_loss = criterion(val_outputs, output_tensor)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')
train(model, input_tensor, output_tensor)


model.eval()
with torch.no_grad():
    predicted_output = model(input_tensor, output_tensor)

input_np = input_tensor.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
output_np = output_tensor.numpy()
predicted_np = predicted_output.numpy()

# Gaussian smoothing
smoothed_predicted_np = gaussian_smooth_and_normalize(predicted_np[0])
smoothed_output_np = gaussian_smooth_and_normalize(output_np[0])
smoothed_input_np = gaussian_smooth_and_normalize(input_np[0])
pdb.set_trace()
# Visualization
def visualize_layers(layers, title, cmap='viridis'):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(layers[i], cmap=cmap)
        ax.set_title(f'Layer {chr(65+i)}')
        plt.colorbar(im, ax=ax)
    fig.suptitle(title)
    plt.show()

output_dir = 'trainingVis'
os.makedirs(output_dir, exist_ok=True)


layer_colors = ['red', 'orange', 'yellow', 'blue', 'green']
layer_names = ['Layer A', 'Layer B', 'Layer C', 'Layer D', 'Layer E']
alpha_values = [1.0, 1.0, 0.9, 0.8, 0.8]

def get_combined_layer(data, end_layer):
    combined_layer = np.sum(data[:end_layer+1], axis=0)
    return combined_layer

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for ax, data, title in zip(
    axes[0],
    [input_np[0], output_np[0], predicted_np[0]],
    ['Gold Layer', 'Original Silver Layer', 'Predicted Silver Layer']
):
    for i in reversed(range(5)):  # Plot E (bottom) to A (top)
        combined_layer = get_combined_layer(data, i)
        X, Y = np.meshgrid(np.arange(combined_layer.shape[0]), np.arange(combined_layer.shape[1]))
        contour = ax.contour(X, Y, combined_layer, levels=10, colors=[layer_colors[i]], alpha=alpha_values[i])
    ax.set_title(title)
    ax.axis('off')

for ax, data, title in zip(
    axes[1],
    [smoothed_input_np, smoothed_output_np, smoothed_predicted_np],
    ['Smoothed Gold Layer', 'Smoothed Silver Layer', 'Smoothed Predicted Silver Layer']
):
    for i in reversed(range(5)):  # E at bottom
        combined_layer = get_combined_layer(data, i)
        X, Y = np.meshgrid(np.arange(combined_layer.shape[0]), np.arange(combined_layer.shape[1]))
        contour = ax.contour(X, Y, combined_layer, levels=10, colors=[layer_colors[i]], alpha=alpha_values[i])
    ax.set_title(title)
    ax.axis('off')

handles = [plt.Line2D([0, 1], [0, 1], color=color, lw=4, alpha=alpha) for color, alpha in zip(layer_colors, alpha_values)]
labels = layer_names

axes[1, -1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig(os.path.join(output_dir, 'contour_stacked_comparison_all_layers.png'))
plt.show()



''' Try to fill in contour plots
layer_colors_bgr = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
layer_colors_rgb = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0)]
layer_names = ['Layer A', 'Layer B', 'Layer C', 'Layer D', 'Layer E']

output_dir = 'trainingVis'
os.makedirs(output_dir, exist_ok=True)

def fill_contours(data, color):
    img = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    contours, _ = cv2.findContours(norm_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=cv2.FILLED)
    return img

def create_filled_contour_images(layers):
    combined_img = np.zeros((layers.shape[1], layers.shape[2], 3), dtype=np.uint8)
    for i in reversed(range(5)):  # Plot E (bottom) to A (top)
        filled_img = fill_contours(layers[i], layer_colors_bgr[i])
        mask = filled_img.any(axis=-1)
        combined_img[mask] = filled_img[mask]
    return combined_img

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

original_images = [
    create_filled_contour_images(input_np[0]),
    create_filled_contour_images(output_np[0]),
    create_filled_contour_images(predicted_np[0])
]

for ax, img, title in zip(axes[0], original_images, ['Gold Layer', 'Original Silver Layer', 'Predicted Silver Layer']):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')

smoothed_images = [
    create_filled_contour_images(smoothed_input_np),
    create_filled_contour_images(smoothed_output_np),
    create_filled_contour_images(smoothed_predicted_np)
]

for ax, img, title in zip(axes[1], smoothed_images, ['Smoothed Gold Layer', 'Smoothed Silver Layer', 'Smoothed Predicted Silver Layer']):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')

handles = [plt.Line2D([0, 1], [0, 1], color=np.array(color)/255, lw=4) for color in layer_colors_rgb]
labels = layer_names

axes[1, -1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig(os.path.join(output_dir, 'contourf_stacked_comparison_all_layers.png'))
plt.show()
'''