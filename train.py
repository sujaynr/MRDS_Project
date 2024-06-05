# train.py
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from models import MineralTransformer


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

# Reshape the input and output to be 3D: (batch_size, sequence_length, feature_dimension)
batch_size = input_layers.shape[0]
sequence_length = input_layers.shape[1]
feature_dimension = grid_size * grid_size

input_layers = input_layers.reshape(batch_size, sequence_length, feature_dimension)
output_layers = output_layers.reshape(batch_size, sequence_length, feature_dimension)

input_tensor = torch.tensor(input_layers, dtype=torch.float32)
output_tensor = torch.tensor(output_layers, dtype=torch.float32).view(batch_size, sequence_length, grid_size, grid_size)

model = MineralTransformer(d_model=d_model)

def train(model, input_tensor, output_tensor, num_epochs=50, learning_rate=0.001):
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
smoothed_output_np = gaussian_smooth_and_normalize(predicted_np[0])

# Visualization
def visualize_layers(layers, title, cmap='viridis'):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(layers[i], cmap=cmap)
        ax.set_title(f'Layer {chr(65+i)}')
        plt.colorbar(im, ax=ax)
    fig.suptitle(title)
    plt.show()

visualize_layers(input_np[0], 'Original Gold Layers (Input)')
visualize_layers(output_np[0], 'Original Silver Layers (Output)')

visualize_layers(predicted_np[0], 'Predicted Silver Layers (Output)')

visualize_layers(smoothed_output_np, 'Smoothed and Normalized Predicted Silver Layers')

output_dir = 'trainingVis'
os.makedirs(output_dir, exist_ok=True)

for i in range(5):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(input_np[0][i], cmap='viridis')
    axes[0].set_title(f'Gold Layer {chr(65+i)} (Input)')
    
    axes[1].imshow(output_np[0][i], cmap='viridis')
    axes[1].set_title(f'Original Silver Layer {chr(65+i)}')
    
    axes[2].imshow(predicted_np[0][i], cmap='viridis')
    axes[2].set_title(f'Predicted Silver Layer {chr(65+i)}')
    
    axes[3].imshow(smoothed_output_np[i], cmap='viridis')
    axes[3].set_title(f'Smoothed Predicted Silver Layer {chr(65+i)}')
    
    plt.savefig(os.path.join(output_dir, f'comparison_layer_{chr(65+i)}.png'))
    plt.close()