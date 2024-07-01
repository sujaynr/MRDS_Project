import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import pdb
import wandb
import segmentation_models_pytorch as smp

from models import LinToConv, SimplifiedMLP, LinToTransformer, MineralDataset, UNet, TransformerToConv
from utils import plot_predictions, integral_loss, evaluate, train

grid_size = 50  # Adjusted to 50x50 grid
hidden_dim = 256  # Adjust as necessary
intermediate_dim = 512  # New intermediate dimension for additional layers
num_minerals = 10
nhead = 4  # Number of heads in the multiheadattention models
num_layers = 1  # Number of layers in the transformer
d_model = 256
dropout_rate = 0.2

# Initialize wandb
wandb.init(project="mineral_transformer_project", name="2048_0.0001TEST")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from HDF5 file
h5_file_path = 'prepared_data_TILES/NEmineral_data.h5'

# Load counts data to find non-empty squares
with h5py.File(h5_file_path, 'r') as f:
    counts = f['counts'][:]

print(f"Original shape of dataset: {counts.shape}")

# Ensure the square is non-empty overall
non_empty_indices = np.where(np.sum(counts, axis=(1, 2, 3)) > 0)[0]
np.random.shuffle(non_empty_indices)
non_empty_indices = np.sort(non_empty_indices)


num_samples = len(non_empty_indices)
num_test_samples = int(num_samples * 0.1)
test_indices = non_empty_indices[:num_test_samples]
train_indices = non_empty_indices[num_test_samples:]  # Use the rest for training

print(f"Shape of dataset after removing empty squares: {counts[non_empty_indices].shape}")

elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
output_mineral_name = 'Nickel'
output_mineral = elements.index(output_mineral_name)
input_minerals = [i for i in range(len(elements)) if i != output_mineral]

print("Layer mapping to elements:")
for i, element in enumerate(elements):
    print(f"Layer {i}: {element}")

# Initialize the dataset and dataloader
train_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=train_indices, train=True)
test_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=test_indices, train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model and train
input_dim = num_minerals * grid_size * grid_size
in_channels = 10 # Input channels (including the masked layer)
out_channels = 1 # Output channel (the infilled layer)

model_type = "LinToConv"  # Change to "UNet", "SimplifiedMLP", "LinToConv", or "LinToTransformer"

if model_type == "TransformerToConv":
    model = TransformerToConv(input_dim, hidden_dim, intermediate_dim, d_model, nhead, num_layers, dropout_rate)
elif model_type == "UNet":
    model = UNet(in_channels, out_channels)
elif model_type == "SimplifiedMLP":
    model = SimplifiedMLP(input_dim, hidden_dim)
elif model_type == "LinToConv":
    model = LinToConv(input_dim, hidden_dim, intermediate_dim)
elif model_type == "LinToTransformer":
    model = LinToTransformer(input_dim, hidden_dim, intermediate_dim, d_model, nhead, num_layers, dropout_rate)

model = model.to(device)
pdb.set_trace()
first_loss = 'integral'
second_loss = 'pixel'
two_step = True

model_type_and_losses = model_type + "_" + first_loss + "_" + second_loss + "_twoStep=" + str(two_step)
predicted_output_test, output_tensor_test = train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.00001, two_step=two_step, first_loss=first_loss, second_loss=second_loss)

# Crop the 64x64 outputs back to 50x50
crop_size = (50, 50)
predicted_output_test = predicted_output_test[:, :, 0:crop_size[0], 0:crop_size[1]]
output_tensor_test = output_tensor_test[:, :, 0:crop_size[0], 0:crop_size[1]]

# After the training process, reshape tensors back to the original shape for visualization and metric computation
batch_size = predicted_output_test.shape[0]
predicted_np_test = predicted_output_test.cpu().numpy().reshape(batch_size, 1, crop_size[0], crop_size[1])
output_np_test = output_tensor_test.cpu().numpy().reshape(batch_size, 1, crop_size[0], crop_size[1])

print(f"Predicted Test Shape: {predicted_np_test.shape}")
print(f"Output Test Shape: {output_np_test.shape}")



# Plot the predictions
plot_predictions(predicted_np_test, output_np_test, counts[test_indices], input_minerals, elements[output_mineral], num_samples=20, specs=model_type_and_losses)


print("Prediction visualizations saved in the 'predictionVis' folder.")
