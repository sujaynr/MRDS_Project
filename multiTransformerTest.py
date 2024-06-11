import os
import pickle
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the conda environment
conda_env = '/Users/sujaynair/anaconda3/envs/dataAnalysis'
os.environ['CONDA_PREFIX'] = conda_env

grid_size = 30
d_model = 512
use_sticky_markov = True
use_validation = False
same_scale = False

def gaussian_smooth_and_normalize(layers, sigma=1.0):
    smoothed_layers = []
    for layer in layers:
        smoothed_layer = gaussian_filter(layer, sigma=sigma)
        normalized_layer = smoothed_layer * (layer.sum() / smoothed_layer.sum())
        smoothed_layers.append(normalized_layer)
    return np.stack(smoothed_layers, axis=0)

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

input_tensor_train = torch.tensor(input_layers, dtype=torch.float32)
output_tensor_train = torch.tensor(output_layers, dtype=torch.float32)

print("Train input tensor shape:", input_tensor_train.shape)
print("Train output tensor shape:", output_tensor_train.shape)

class MineralTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2):
        super(MineralTransformer, self).__init__()
        self.input_projection = nn.Linear(grid_size * grid_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, grid_size * grid_size)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        batch_size, seq_length, feature_dim = src.shape
        
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        output = self.output_projection(output)
        return output.view(batch_size, seq_length, grid_size, grid_size)

model = MineralTransformer(d_model=d_model)

def train(model, input_tensor_train, output_tensor_train, num_epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor_train, output_tensor_train)
        loss = criterion(outputs.view(-1, grid_size * grid_size), output_tensor_train.view(-1, grid_size * grid_size))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping, maybe not necessary
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        pdb.set_trace()

train(model, input_tensor_train, output_tensor_train)

model.eval()
with torch.no_grad():
    predicted_output_train = model(input_tensor_train, output_tensor_train)

input_np_train = input_tensor_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
output_np_train = output_tensor_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)
predicted_np_train = predicted_output_train.numpy().reshape(batch_size, sequence_length, grid_size, grid_size)

# Gaussian smoothing
smoothed_predicted_np_train = gaussian_smooth_and_normalize(predicted_np_train[0])
smoothed_output_np_train = gaussian_smooth_and_normalize(output_np_train[0])
smoothed_input_np_train = gaussian_smooth_and_normalize(input_np_train[0])

#VIS
def visualize_layers(layer_index, input_data, output_data, predicted_data, input_elements, output_elements):
    fig, axes = plt.subplots(3, max(len(input_elements), len(output_elements)), figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Input
    for j in range(len(input_elements)):
        ax = axes[0, j]
        im = ax.imshow(input_data[j][layer_index], cmap='viridis')
        ax.set_title(f'{input_elements[j]} Layer {chr(65+layer_index)}')
        plt.colorbar(im, ax=ax)
    
    # GT
    for k in range(len(output_elements)):
        ax = axes[1, k]
        im = ax.imshow(output_data[k][layer_index], cmap='viridis')
        ax.set_title(f'{output_elements[k]} Layer {chr(65+layer_index)}')
        plt.colorbar(im, ax=ax)
    
    # Predictions
    for l in range(len(output_elements)):
        ax = axes[2, l]
        im = ax.imshow(predicted_data[l][layer_index], cmap='viridis')
        ax.set_title(f'Predicted {output_elements[l]} Layer {chr(65+layer_index)}')
        plt.colorbar(im, ax=ax)
    
    fig.suptitle(f'Comparison for Quality Layer {chr(65+layer_index)}', fontsize=16)
    plt.savefig(os.path.join('trainingVis', f'comparison_layer_{chr(65+layer_index)}.png'))
    plt.show()


for i in range(5): # By quality
    visualize_layers(i, input_np_train, output_np_train, predicted_np_train, input_elements, output_elements)

# Metric
for i in range(5):
    predicted_sum = np.sum(predicted_np_train[0][i])
    ground_truth_sum = np.sum(output_np_train[0][i])
    metric = predicted_sum / ground_truth_sum
    print(f'Layer {chr(65+i)}: Metric = {metric}')