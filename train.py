import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import random
import geopandas as gpd

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
parser = argparse.ArgumentParser(description="Train a model for mineral prediction.")
parser.add_argument('--grid_size', type=int, default=50, help='Grid size of the input data.')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the models.')
parser.add_argument('--intermediate_dim', type=int, default=512, help='Intermediate dimension for additional layers.')
parser.add_argument('--num_minerals', type=int, default=11, help='Number of dimensions.')
parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the multihead attention models.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the transformer.')
parser.add_argument('--d_model', type=int, default=256, help='Model dimension for the transformer.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the models.')
parser.add_argument('--model_type', type=str, default='u', choices=['tc', 'u', 'lc', 'lt', 'l'], help='Model type: tc (TransformerToConv), u (UNet), lc (LinToConv), lt (LinToTransformer), l (SimplifiedMLP).')
parser.add_argument('--output_mineral_name', type=str, default='Nickel', help='Name of the output mineral.')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for training.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--loss1', type=str, default='integral', help='First loss function.')
parser.add_argument('--two_step', type=str, default=True, help='Whether to use two-step training.')
parser.add_argument('--logName', type=str, default='testB2', help='Name of the log file.')
parser.add_argument('--use_bce', action='store_true', help='Use binary cross-entropy loss for resource presence detection.')

args = parser.parse_args()

# Configuration
grid_size = args.grid_size
hidden_dim = args.hidden_dim
intermediate_dim = args.intermediate_dim
num_minerals = args.num_minerals
nhead = args.nhead
num_layers = args.num_layers
d_model = args.d_model
dropout_rate = args.dropout_rate
model_type = args.model_type
output_mineral_name = args.output_mineral_name
learning_rate = args.learning_rate
num_epochs = args.num_epochs
first_loss = args.loss1
two_step = (args.two_step == 'True')
use_bce = args.use_bce

print("Configuration Summary:")
print(f"  Grid Size: {grid_size}")
print(f"  Hidden Dimension: {hidden_dim}")
print(f"  Intermediate Dimension: {intermediate_dim}")
print(f"  Number of Minerals: {num_minerals}")
print(f"  Number of Heads: {nhead}")
print(f"  Number of Layers: {num_layers}")
print(f"  Model Dimension: {d_model}")
print(f"  Dropout Rate: {dropout_rate}")
print(f"  Model Type: {model_type}")
print(f"  Output Mineral Name: {output_mineral_name}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Number of Epochs: {num_epochs}")
print(f"  First Loss Function: {first_loss}")
print(f"  Two-Step Training: {two_step}")
print(f"  Use BCE Loss: {use_bce}")

# Set log name:
model_type_and_losses = model_type + "_" + output_mineral_name + "_firstLoss=" + first_loss + "_twoStep=" + str(two_step) + "_lr=" + str(args.learning_rate) + "_" + args.logName
config = {
    "grid_size": grid_size,
    "hidden_dim": hidden_dim,
    "intermediate_dim": intermediate_dim,
    "num_minerals": num_minerals,
    "nhead": nhead,
    "num_layers": num_layers,
    "d_model": d_model,
    "dropout_rate": dropout_rate,
    "model_type": model_type,
    "output_mineral_name": output_mineral_name,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "first_loss": first_loss,
    "two_step": two_step,
    "use_bce": use_bce
}

# Initialize wandb
wandb.init(project="mineral_transformer_project", name=model_type_and_losses, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from HDF5 file
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
fault_file_path = 'prepared_data_TILES/faultData.h5'

with h5py.File(h5_file_path, 'r') as f:
    coords = f['coordinates'][:]
    counts = f['counts'][:]
    qualities = f['qualities'][:]
with h5py.File(fault_file_path, 'r') as f:
    fault_data = f['faults'][:]

fault_slice = fault_data[:, 0, :, :]
fault_slice_expanded = np.expand_dims(fault_slice, axis=1)
counts = np.concatenate((counts, fault_slice_expanded), axis=1)
print(f"Counts shape after adding faults: {counts.shape}")
counts = np.nan_to_num(counts, nan=-10)



binary_counts = (counts > 0).astype(np.float32)


num_samples = len(counts)
num_test_samples = int(num_samples * 0.1)
test_indices = np.arange(num_test_samples)
train_indices = np.arange(num_test_samples, num_samples)


shapefile_path = '/home/sujaynair/ne_110m_admin_0_countries.shp'
output_path = '/home/sujaynair/MRDS_Project/tilingVIS/trainValVIS.png'
us_shapefile = gpd.read_file(shapefile_path)
us_shape = us_shapefile[us_shapefile['ADMIN'] == 'United States of America']
def miles_to_degrees_lat(miles):
    return miles / 69.0

def miles_to_degrees_lon(miles, latitude):
    return miles / (69.0 * np.cos(np.radians(latitude)))

# Visualize the squares
def visualize_squares_with_coords(coords, train_indices, test_indices, us_shape, output_path):
    fig, ax = plt.subplots(figsize=(15, 10))
    us_shape.boundary.plot(ax=ax, color='black')
    
    # Plot train squares
    for idx in train_indices:
        lat_start, lon_start = coords[idx]
        circle = plt.Circle((lon_start, lat_start), miles_to_degrees_lon(5, lat_start), color='blue', alpha=0.3, edgecolor='none')
        ax.add_patch(circle)
    
    # Plot test squares
    for idx in test_indices:
        lat_start, lon_start = coords[idx]
        circle = plt.Circle((lon_start, lat_start), miles_to_degrees_lon(5, lat_start), color='red', alpha=0.3, edgecolor='none')
        ax.add_patch(circle)
    
    # Add legend
    train_patch = plt.Circle((0, 0), 1, color='blue', alpha=0.3, edgecolor='none')
    test_patch = plt.Circle((0, 0), 1, color='red', alpha=0.3, edgecolor='none')
    ax.legend([train_patch, test_patch], ['Train Squares', 'Test Squares'], loc='upper right')
    
    plt.title("Train and Test Squares Mapped on US")
    plt.savefig(output_path)
    plt.close(fig)
visualize_squares_with_coords(coords, train_indices, test_indices, us_shape, output_path)
# assert(False)







elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
output_mineral = elements.index(output_mineral_name)
input_minerals = [i for i in range(len(elements)) if i != output_mineral]



print("Layer mapping to elements:")
for i, element in enumerate(elements):
    print(f"Layer {i}: {element}")

print(f"Output Mineral: {output_mineral_name} (Layer {output_mineral})")
use_unet_padding = False
if model_type == "u":
    use_unet_padding = True
if use_bce:
    train_dataset = MineralDataset(binary_counts, input_minerals, output_mineral, indices=train_indices, train=True, unet=use_unet_padding)
    test_dataset = MineralDataset(binary_counts, input_minerals, output_mineral, indices=test_indices, train=False, unet=use_unet_padding)
else:
    train_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=train_indices, train=True, unet=use_unet_padding)
    test_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=test_indices, train=False, unet=use_unet_padding)



train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



if model_type == "tc":
    model = TransformerToConv(input_dim=num_minerals * grid_size * grid_size, hidden_dim=hidden_dim, 
                              intermediate_dim=intermediate_dim, d_model=d_model, nhead=nhead, 
                              num_layers=num_layers, dropout_rate=dropout_rate)
elif model_type == "u":
    model = UNet(in_channels=num_minerals, out_channels=1)
elif model_type == "lc":
    model = LinToConv(input_dim=num_minerals * grid_size * grid_size, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
elif model_type == "lt":
    model = LinToTransformer(input_dim=num_minerals * grid_size * grid_size, hidden_dim=hidden_dim, 
                             intermediate_dim=intermediate_dim, d_model=d_model, nhead=nhead, 
                             num_layers=num_layers, dropout_rate=dropout_rate)
elif model_type == "l":
    model = SimplifiedMLP(input_dim=num_minerals * grid_size * grid_size, hidden_dim=hidden_dim)
else:
    raise ValueError(f"Unknown model type: {model_type}")

model = model.to(device)
# pdb.set_trace()

if use_bce:
    criterion = nn.BCEWithLogitsLoss()  # BCE with logits
else:
    if first_loss == "integral":
        criterion = integral_loss
    else:
        criterion = regular_loss

predicted_output_test, output_tensor_test = train(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion, two_step=two_step, first_loss=first_loss, use_bce=use_bce)
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
plot_predictions(predicted_np_test, output_np_test, counts[test_indices], input_minerals, fault_data[test_indices], elements[output_mineral], num_samples=20, specs=model_type_and_losses)

print("Prediction visualizations saved in the 'predictionVis' folder.")
