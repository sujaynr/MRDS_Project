import os
import pickle
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from models import MineralTransformer

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the conda environment
conda_env = '/Users/sujaynair/anaconda3/envs/dataAnalysis'
os.environ['CONDA_PREFIX'] = conda_env

grid_size = 30
d_model = 512
use_sticky_markov = True  # Set this flag to use the 2D Sticky Markov model for train/val split

def gaussian_smooth_and_normalize(layers, sigma=1.0):
    smoothed_layers = []
    for layer in layers:
        smoothed_layer = gaussian_filter(layer, sigma=sigma)
        normalized_layer = smoothed_layer * (layer.sum() / smoothed_layer.sum())
        smoothed_layers.append(normalized_layer)
    return np.stack(smoothed_layers, axis=0)

def generate_sticky_markov_mask(grid_size, train_ratio=0.7):
    np.random.seed(0)  # For reproducibility
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    start_train = np.random.choice([True, False], size=(1,))[0]
    current_state = start_train
    for i in range(grid_size):
        for j in range(grid_size):
            if np.random.rand() < 0.9:  # High probability to stay in the current state
                mask[i, j] = current_state
            else:
                current_state = not current_state
                mask[i, j] = current_state
    if mask.sum() / mask.size < train_ratio:
        mask = ~mask
    # Apply Gaussian blur to the mask
    mask = gaussian_filter(mask.astype(float), sigma=1.0)
    # Set fractional threshold for inclusion/exclusion
    threshold = np.percentile(mask, train_ratio * 100)
    mask = mask >= threshold
    return mask.flatten()

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
batch_size = input_layers.shape[0]
sequence_length = input_layers.shape[1]
feature_dimension = grid_size * grid_size

input_layers = input_layers.reshape(batch_size, sequence_length, feature_dimension)
output_layers = output_layers.reshape(batch_size, sequence_length, feature_dimension)

print("Reshaped input layers shape:", input_layers.shape)
print("Reshaped output layers shape:", output_layers.shape)

# Create train/validation mask
total_cells = grid_size * grid_size

if use_sticky_markov:
    mask = generate_sticky_markov_mask(grid_size)
    train_mask = mask
    val_mask = ~mask
else:
    indices = np.arange(total_cells)
    np.random.shuffle(indices)
    train_indices = indices[:int(0.7 * total_cells)] # Add command line arg for this eventually
    val_indices = indices[int(0.7 * total_cells):]

    train_mask = np.zeros(total_cells, dtype=bool)
    train_mask[train_indices] = True
    val_mask = ~train_mask

input_train = np.zeros_like(input_layers)
output_train = np.zeros_like(output_layers)
input_val = np.zeros_like(input_layers)
output_val = np.zeros_like(output_layers)

input_train[:, :, train_mask] = input_layers[:, :, train_mask]
output_train[:, :, train_mask] = output_layers[:, :, train_mask]
input_val[:, :, val_mask] = input_layers[:, :, val_mask]
output_val[:, :, val_mask] = output_layers[:, :, val_mask]

input_tensor_train = torch.tensor(input_train, dtype=torch.float32)
output_tensor_train = torch.tensor(output_train, dtype=torch.float32)
input_tensor_val = torch.tensor(input_val, dtype=torch.float32)
output_tensor_val = torch.tensor(output_val, dtype=torch.float32)

print("Train input tensor shape:", input_tensor_train.shape)
print("Train output tensor shape:", output_tensor_train.shape)
print("Val input tensor shape:", input_tensor_val.shape)
print("Val output tensor shape:", output_tensor_val.shape)

model = MineralTransformer(d_model=d_model)

def train(model, input_tensor_train, output_tensor_train, input_tensor_val, output_tensor_val, num_epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor_train, output_tensor_train)
        loss = criterion(outputs.view(-1, grid_size * grid_size), output_tensor_train.view(-1, grid_size * grid_size))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(input_tensor_val, output_tensor_val)
            val_loss = criterion(val_outputs.view(-1, grid_size * grid_size), output_tensor_val.view(-1, grid_size * grid_size))

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

train(model, input_tensor_train, output_tensor_train, input_tensor_val, output_tensor_val)

output_dir = 'trainingVis'
os.makedirs(output_dir, exist_ok=True)

layer_colors = ['red', 'orange', 'yellow', 'blue', 'green']
layer_names = ['Layer A', 'Layer B', 'Layer C', 'Layer D', 'Layer E']
alpha_values = [1.0, 1.0, 0.9, 0.8, 0.8]

def get_combined_layer(data, end_layer):
    combined_layer = np.sum(data[:end_layer+1], axis=0)
    return combined_layer

model.eval()
with torch.no_grad():
    predicted_output_train = model(input_tensor_train, output_tensor_train)
    predicted_output_val = model(input_tensor_val, output_tensor_val)

input_np_train = input_tensor_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
output_np_train = output_tensor_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
predicted_np_train = predicted_output_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)

input_np_val = input_tensor_val.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
output_np_val = output_tensor_val.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
predicted_np_val = predicted_output_val.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)

# Gaussian smoothing
smoothed_predicted_np_train = gaussian_smooth_and_normalize(predicted_np_train[0])
smoothed_output_np_train = gaussian_smooth_and_normalize(output_np_train[0])
smoothed_input_np_train = gaussian_smooth_and_normalize(input_np_train[0])

smoothed_predicted_np_val = gaussian_smooth_and_normalize(predicted_np_val[0])
smoothed_output_np_val = gaussian_smooth_and_normalize(output_np_val[0])
smoothed_input_np_val = gaussian_smooth_and_normalize(input_np_val[0])

# Visualization
def visualize_layers(layers, title, cmap='viridis'):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(layers[i], cmap=cmap)
        ax.set_title(f'Layer {chr(65+i)}')
        plt.colorbar(im, ax=ax)
    fig.suptitle(title)
    plt.show()

visualize_layers(input_np_train[0], 'Train Original Gold Layers (Input)')
visualize_layers(output_np_train[0], 'Train Original Silver Layers (Output)')
visualize_layers(predicted_np_train[0], 'Train Predicted Silver Layers (Output)')

visualize_layers(input_np_val[0], 'Val Original Gold Layers (Input)')
visualize_layers(output_np_val[0], 'Val Original Silver Layers (Output)')
visualize_layers(predicted_np_val[0], 'Val Predicted Silver Layers (Output)')

fig, axes = plt.subplots(5, 4, figsize=(20, 25))

titles = ['Train Original Silver', 'Train Predicted Silver', 'Val Original Silver', 'Val Predicted Silver']
data_sets = [output_np_train[0], predicted_np_train[0], output_np_val[0], predicted_np_val[0]]

for i in range(5):
    for ax, data, title in zip(axes[i], data_sets, titles):
        im = ax.imshow(data[i], cmap='viridis')
        ax.set_title(f'{title} Layer {chr(65+i)}')
        plt.colorbar(im, ax=ax)
        
plt.savefig(os.path.join(output_dir, 'SM_comparison_all_predicted_silver_layers.png'))
plt.show()


''' OLD CONTOUR PLOTTING
# Visualization code remains the same
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
    [input_np_train[0], output_np_train[0], predicted_np_train[0]],
    [f'{input_elements[0]} Layer', f'Original {output_elements[0]} Layer', f'Predicted {output_elements[0]} Layer']
):
    for i in reversed(range(5)):  # Plot E (bottom) to A (top)
        combined_layer = get_combined_layer(data, i)
        X, Y = np.meshgrid(np.arange(combined_layer.shape[0]), np.arange(combined_layer.shape[1]))
        contour = ax.contour(X, Y, combined_layer, levels=10, colors=[layer_colors[i]], alpha=alpha_values[i])
    ax.set_title(title)
    ax.axis('off')

for ax, data, title in zip(
    axes[1],
    [smoothed_input_np_train, smoothed_output_np_train, smoothed_predicted_np_train],
    [f'Smoothed {input_elements[0]} Layer', f'Smoothed {output_elements[0]} Layer', f'Smoothed Predicted {output_elements[0]} Layer']
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

plt.savefig(os.path.join(output_dir, f'{input_elements[0]}_to_{output_elements[0]}_contour_stacked_comparison_all_layers.png'))
plt.show()
'''


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