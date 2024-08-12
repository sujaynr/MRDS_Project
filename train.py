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
from utils import plot_predictions, integral_loss, regular_loss, nonempty_loss, combined_loss, dice_coefficient_nonzero, create_nonzero_mask, masked_mse_loss, absolute_difference_integral


parser = argparse.ArgumentParser(description="Train a model for mineral prediction.")
parser.add_argument('--grid_size', type=int, default=50, help='Grid size of the input data.')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the models.')
parser.add_argument('--intermediate_dim', type=int, default=512, help='Intermediate dimension for additional layers.')
parser.add_argument('--num_minerals', type=int, default=15, help='Number of dimensions.')
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
parser.add_argument('--logName', type=str, default='testB3', help='Name of the log file.')
parser.add_argument('--use_bce', action='store_true', help='Use binary cross-entropy loss for resource presence detection.')
parser.add_argument('--tn', action='store_true', help='Include true negatives in the evaluation.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--set_seed', type=int, default=42, help='Set seed for reproducibility.')

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
tn = args.tn
batch_size = args.batch_size
set_seed = args.set_seed

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
print(f"  Include True Negatives: {tn}")
print(f"  Batch Size: {batch_size}")
print(f"  Set Seed: {set_seed}")

# Set log name:
lognameoutput = args.logName
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
    "use_bce": use_bce,
    "include_true_negatives": tn,
    "batch_size": batch_size,
    "set_seed": set_seed
}

# python train.py --grid_size 50 --hidden_dim 256 --intermediate_dim 512 --num_minerals 11 --nhead 4 --num_layers 1 --d_model 256 --dropout_rate 0.2 --model_type u --output_mineral_name Nickel --learning_rate 0.00001 --num_epochs 10 --loss1 integral --two_step True --logName testB2 --use_bce

torch.manual_seed(set_seed)
np.random.seed(set_seed)
random.seed(set_seed)

# Initialize wandb
wandb.init(project="mineral_transformer_project", name=lognameoutput, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from HDF5 file
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
fault_file_path = 'prepared_data_TILES/faultData.h5'
geoAge_file_path = 'prepared_data_TILES/geoAge.h5'
elevation_file_path = '/home/sujaynair/MRDS_Project/all_elevations.h5'

with h5py.File(h5_file_path, 'r') as f:
    coords = f['coordinates'][:]
    counts = f['counts'][:]
    qualities = f['qualities'][:]
with h5py.File(fault_file_path, 'r') as f:
    fault_data = f['faults'][:]
with h5py.File(geoAge_file_path, 'r') as f:
    geoAge_data = f['geoinfo'][:]
with h5py.File(elevation_file_path, 'r') as f:
    elevations = f['elevations'][:]

############################################
def normalize_layer(layer):
    min_val = np.min(layer)
    max_val = np.max(layer)
    return (layer - min_val) / (max_val - min_val)

# Extract minage and maxage layers
min_age_layer = geoAge_data[:, 0, :, :]
max_age_layer = geoAge_data[:, 1, :, :]

# Normalize minage and maxage layers
normalized_min_age_layer = normalize_layer(min_age_layer)
normalized_max_age_layer = normalize_layer(max_age_layer)

# Replace the original layers with the normalized layers
geoAge_data[:, 0, :, :] = normalized_min_age_layer
geoAge_data[:, 1, :, :] = normalized_max_age_layer

normalized_elevations = normalize_layer(elevations)

# Concatenate fault data and normalized geoAge data to counts
fault_slice = fault_data[:, 0, :, :]
fault_slice_expanded = np.expand_dims(fault_slice, axis=1)
elevation_slice_expanded = np.expand_dims(normalized_elevations, axis=1)


counts = np.concatenate((counts, fault_slice_expanded), axis=1)
print(f"Counts shape after adding faults: {counts.shape}")
counts = np.nan_to_num(counts, nan=-10)
counts = np.concatenate((counts, geoAge_data), axis=1)
print(f"Counts shape after adding geoAge_data: {counts.shape}")
counts = np.concatenate((counts, elevation_slice_expanded), axis=1)
print(f"Counts shape after adding elevation data: {counts.shape}")
########################################################
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
# visualize_squares_with_coords(coords, train_indices, test_indices, us_shape, output_path)
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



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



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
def train(model, train_loader, test_loader, num_epochs=50, learning_rate=0.0001, criterion=integral_loss, two_step=False, first_loss='integral', use_bce=False, include_true_negatives=False):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    if use_bce:
        criterion = nn.BCEWithLogitsLoss()
    else:
        if first_loss == 'integral':
            criterion = integral_loss
        else:
            criterion = regular_loss
    
    losses = []
    test_losses = []
    nonempty_losses = []
    diff_integrals = []
    dice_coefs = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_nonempty_loss = 0
        total_diff_integral = 0
        total_dice_coef = 0
        num_batches = 0

        for input_tensor_train, output_tensor_train in train_loader:
            input_tensor_train.requires_grad = True
            output_tensor_train.requires_grad = True
            optimizer.zero_grad()

            outputs = model(input_tensor_train.to(device))

            if use_bce:
                loss = criterion(outputs, output_tensor_train.to(device))
            else:
                if first_loss == 'integral':
                    loss = criterion(outputs, output_tensor_train.to(device))
                else:
                    mask = create_nonzero_mask(output_tensor_train).to(device)
                    loss = masked_mse_loss(outputs, output_tensor_train, mask, include_true_negatives=include_true_negatives)

            nonempty_loss_value = nonempty_loss(outputs, output_tensor_train.to(device))
            diff_integral = absolute_difference_integral(outputs, output_tensor_train).item()
            dice_coef = dice_coefficient_nonzero(outputs, output_tensor_train).item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nonempty_loss += nonempty_loss_value.item()
            total_diff_integral += diff_integral
            total_dice_coef += dice_coef
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_nonempty_loss = total_nonempty_loss / num_batches
        avg_diff_integral = total_diff_integral / num_batches
        avg_dice_coef = total_dice_coef / num_batches

        losses.append(avg_loss)
        test_loss, nonempty_test_loss, _, _, test_diff_integral, test_dice_coef = evaluate(model, test_loader, criterion, use_bce=use_bce, include_true_negatives=include_true_negatives)
        test_losses.append(test_loss)
        nonempty_losses.append(nonempty_test_loss)
        diff_integrals.append(test_diff_integral)
        dice_coefs.append(test_dice_coef)

        wandb.log({
            "Train Loss (Step 1)": avg_loss, 
            "Test Loss (Step 1)": test_loss, 
            "Non-Empty Train Loss (Step 1)": avg_nonempty_loss, 
            "Non-Empty Test Loss (Step 1)": nonempty_test_loss,
            "Difference in Integrals (Train)": avg_diff_integral,
            "Difference in Integrals (Test)": test_diff_integral,
            "Dice Coefficient (Train)": avg_dice_coef,
            "Dice Coefficient (Test)": test_dice_coef
        })
        print(f'Epoch {epoch+1}/{num_epochs} (Step 1), Train Loss: {avg_loss}, Test Loss: {test_loss}, Non-Empty Train Loss: {avg_nonempty_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Train): {avg_diff_integral}, Difference in Integrals (Test): {test_diff_integral}, Dice Coefficient (Train): {avg_dice_coef}, Dice Coefficient (Test): {test_dice_coef}')

    if two_step:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_nonempty_loss = 0
            total_diff_integral = 0
            total_dice_coef = 0
            num_batches = 0

            for input_tensor_train, output_tensor_train in train_loader:
                input_tensor_train.requires_grad = True
                optimizer.zero_grad()
                outputs = model(input_tensor_train.to(device))

                if use_bce:
                    loss = criterion(outputs, output_tensor_train.to(device))
                else:
                    loss = combined_loss(outputs, output_tensor_train, first_loss, include_true_negatives=include_true_negatives)
                
                nonempty_loss_value = nonempty_loss(outputs, output_tensor_train.to(device))
                diff_integral = absolute_difference_integral(outputs, output_tensor_train).item()
                dice_coef = dice_coefficient_nonzero(outputs, output_tensor_train).item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_nonempty_loss += nonempty_loss_value.item()
                total_diff_integral += diff_integral
                total_dice_coef += dice_coef
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_nonempty_loss = total_nonempty_loss / num_batches
            avg_diff_integral = total_diff_integral / num_batches
            avg_dice_coef = total_dice_coef / num_batches

            losses.append(avg_loss)
            test_loss, nonempty_test_loss, _, _, test_diff_integral, test_dice_coef = evaluate(model, test_loader, lambda p, t: combined_loss(p, t, first_loss, include_true_negatives=include_true_negatives), use_bce=use_bce, include_true_negatives=include_true_negatives)
            test_losses.append(test_loss)
            nonempty_losses.append(nonempty_test_loss)
            diff_integrals.append(test_diff_integral)
            dice_coefs.append(test_dice_coef)

            wandb.log({
                "Train Loss (Step 2)": avg_loss, 
                "Test Loss (Step 2)": test_loss, 
                "Non-Empty Train Loss (Step 2)": avg_nonempty_loss, 
                "Non-Empty Test Loss (Step 2)": nonempty_test_loss,
                "Difference in Integrals (Train)": avg_diff_integral,
                "Difference in Integrals (Test)": test_diff_integral,
                "Dice Coefficient (Train)": avg_dice_coef,
                "Dice Coefficient (Test)": test_dice_coef
            })
            print(f'Epoch {epoch+1}/{num_epochs} (Step 2), Train Loss: {avg_loss}, Test Loss: {test_loss}, Non-Empty Train Loss: {avg_nonempty_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Train): {avg_diff_integral}, Difference in Integrals (Test): {test_diff_integral}, Dice Coefficient (Train): {avg_dice_coef}, Dice Coefficient (Test): {test_dice_coef}')

    final_criterion = lambda p, t: combined_loss(p, t, first_loss, include_true_negatives=include_true_negatives) if two_step else criterion(p, t)
    test_loss, nonempty_test_loss, predicted_output_test, output_tensor_test, final_diff_integral, final_dice_coef = evaluate(model, test_loader, final_criterion, use_bce=use_bce, include_true_negatives=include_true_negatives)
    print(f'Final Test Loss: {test_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Test): {final_diff_integral}, Dice Coefficient (Test): {final_dice_coef}')

    final_train_metrics = compute_metrics(train_loader, model, final_criterion, use_bce, include_true_negatives)
    final_test_metrics = compute_metrics(test_loader, model, final_criterion, use_bce, include_true_negatives)
    wandb.log(final_train_metrics)
    wandb.log(final_test_metrics)

    plt.plot(losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(nonempty_losses, label='Non-Empty Test Loss')
    plt.plot(diff_integrals, label='Difference in Integrals')
    plt.plot(dice_coefs, label='Dice Coefficient')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('trainingVis/loss_curve.png')
    plt.close()

    wandb.save('trainingVis/loss_curve.png')
    return predicted_output_test, output_tensor_test


def evaluate(model, data_loader, criterion, use_bce=False, include_true_negatives=False):
    model.eval()
    total_loss = 0
    total_nonempty_loss = 0
    total_diff_integral = 0
    total_dice_coef = 0
    num_batches = 0
    predicted_output = []
    output_tensor = []

    with torch.no_grad():
        for input_tensor, output in data_loader:
            input_tensor = input_tensor.to(device)
            output = output.to(device)
            outputs = model(input_tensor)

            if use_bce:
                loss = criterion(outputs, output).item()
                outputs = torch.sigmoid(outputs)
            else:
                if include_true_negatives:
                    mask = create_nonzero_mask(output)
                    loss = masked_mse_loss(outputs, output, mask, include_true_negatives=True).item()
                else:
                    loss = criterion(outputs, output).item()

            nonempty_loss_value = nonempty_loss(outputs, output).item()
            diff_integral = absolute_difference_integral(outputs, output).item()
            dice_coef = dice_coefficient_nonzero(outputs, output).item()

            total_loss += loss
            total_nonempty_loss += nonempty_loss_value
            total_diff_integral += diff_integral
            total_dice_coef += dice_coef
            num_batches += 1

            predicted_output.append(outputs)
            output_tensor.append(output)

    avg_loss = total_loss / num_batches
    avg_nonempty_loss = total_nonempty_loss / num_batches
    avg_diff_integral = total_diff_integral / num_batches
    avg_dice_coef = total_dice_coef / num_batches
    predicted_output = torch.cat(predicted_output, dim=0)
    output_tensor = torch.cat(output_tensor, dim=0)

    return avg_loss, avg_nonempty_loss, predicted_output, output_tensor, avg_diff_integral, avg_dice_coef

def compute_metrics(data_loader, model, criterion, use_bce, include_true_negatives):
    model.eval()
    total_dice_coef = 0.0
    total_diff_integral = 0.0
    total_nonempty_mse = 0.0
    total_pixel_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_tensor, output_tensor in data_loader:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            outputs = model(input_tensor)

            dice_coef = dice_coefficient_nonzero(outputs, output_tensor).item()
            diff_integral = absolute_difference_integral(outputs, output_tensor).item()
            nonempty_mse = nonempty_loss(outputs, output_tensor).item()
            pixel_mse = regular_loss(outputs, output_tensor).item()

            total_dice_coef += dice_coef
            total_diff_integral += diff_integral
            total_nonempty_mse += nonempty_mse
            total_pixel_mse += pixel_mse
            num_batches += 1

    avg_dice_coef = total_dice_coef / num_batches
    avg_diff_integral = total_diff_integral / num_batches
    avg_nonempty_mse = total_nonempty_mse / num_batches
    avg_pixel_mse = total_pixel_mse / num_batches

    metrics = {
        "Avg Dice Coefficient": avg_dice_coef,
        "Avg Difference in Integral": avg_diff_integral,
        "Avg Non-Empty MSE": avg_nonempty_mse,
        "Avg Pixel MSE": avg_pixel_mse
    }

    return metrics

if use_bce:
    criterion = nn.BCEWithLogitsLoss()
else:
    if first_loss == "integral":
        criterion = integral_loss
    else:
        criterion = regular_loss

predicted_output_test, output_tensor_test = train(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion, two_step=two_step, first_loss=first_loss, use_bce=use_bce, include_true_negatives=tn)

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

plot_predictions(predicted_np_test, output_np_test, counts[test_indices], input_minerals, fault_data[test_indices], geoAge_data[test_indices], elevations[test_indices], elements[output_mineral], num_samples=20, specs=lognameoutput)
