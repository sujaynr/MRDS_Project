import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import random
import wandb
from torch.utils.data import Dataset, DataLoader
import pdb
import matplotlib.pyplot as plt
import os

from models import LinToConv, SimplifiedMLP, LinToTransformer, MineralDataset, UNet, TransformerToConv
from utils import plot_predictions, integral_loss, regular_loss, nonempty_loss, combined_loss, dice_coefficient_nonzero, create_nonzero_mask, masked_mse_loss, absolute_difference_integral

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for mineral prediction.")
parser.add_argument('--grid_size', type=int, default=50, help='Grid size of the input data.')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the models.')
parser.add_argument('--intermediate_dim', type=int, default=512, help='Intermediate dimension for additional layers.')
parser.add_argument('--num_minerals', type=int, default=15, help='Number of input dimensions (minerals).')
parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the multihead attention models.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the transformer.')
parser.add_argument('--d_model', type=int, default=256, help='Model dimension for the transformer.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the models.')
parser.add_argument('--model_type', type=str, default='u', choices=['tc', 'u', 'lc', 'lt', 'l'], help='Model type.')
parser.add_argument('--output_mineral_name', type=str, default='Nickel', help='Name of the output mineral.')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for training.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--loss1', type=str, default='integral', help='First loss function.')
parser.add_argument('--two_step', type=str, default='True', help='Whether to use two-step training.')
parser.add_argument('--logName', type=str, default='testB3', help='Name of the log file.')
parser.add_argument('--use_bce', action='store_true', help='Use binary cross-entropy loss for resource presence detection.')
parser.add_argument('--tn', action='store_true', help='Include true negatives in the evaluation.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--set_seed', type=int, default=42, help='Set seed for reproducibility.')
parser.add_argument('--use_raca', action='store_true', help='Use RaCA data for training.')

args = parser.parse_args()

# Configuration
torch.manual_seed(args.set_seed)
np.random.seed(args.set_seed)
random.seed(args.set_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = vars(args)
wandb.init(project="mineral_transformer_project", name=args.logName, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to HDF5 files
h5_file_path = 'prepared_data_TILES/mineralDataWithCoords.h5'
fault_file_path = 'prepared_data_TILES/faultData.h5'
geoAge_file_path = 'prepared_data_TILES/geoAge.h5'
elevation_file_path = '/home/sujaynair/MRDS_Project/all_elevations.h5'
raca_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/racagridsFILLED.h5'

# Function to load datasets from HDF5 files
def load_hdf5_data(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

# Function to normalize layers
def normalize_layer(layer):
    min_val = np.min(layer)
    max_val = np.max(layer)
    return (layer - min_val) / (max_val - min_val)

# Load datasets
coords = load_hdf5_data(h5_file_path, 'coordinates')
counts = load_hdf5_data(h5_file_path, 'counts')
fault_data = load_hdf5_data(fault_file_path, 'faults')
geoAge_data = load_hdf5_data(geoAge_file_path, 'geoinfo')
elevations = load_hdf5_data(elevation_file_path, 'elevations')

if args.use_raca:
    # pdb.set_trace()
    raca_data = load_hdf5_data(raca_file_path, 'aggregated_output')
    print(f"Loaded RaCA data with shape: {raca_data.shape}")

# Normalize geoAge and elevation layers
geoAge_data[:, 0, :, :] = normalize_layer(geoAge_data[:, 0, :, :])
geoAge_data[:, 1, :, :] = normalize_layer(geoAge_data[:, 1, :, :])
normalized_elevations = normalize_layer(elevations)

# Expand dimensions for concatenation
fault_slice_expanded = np.expand_dims(fault_data[:, 0, :, :], axis=1)
elevation_slice_expanded = np.expand_dims(normalized_elevations, axis=1)

# Concatenate datasets
counts = np.concatenate((counts, fault_slice_expanded, geoAge_data, elevation_slice_expanded), axis=1)
counts = np.nan_to_num(counts, nan=-10)  # Replace NaNs with a specific value

if args.use_raca:
    raca_data_expanded = np.transpose(raca_data, (0, 3, 1, 2))
    counts = np.concatenate((counts, raca_data_expanded), axis=1)
    print(f"Counts shape after adding RaCA data: {counts.shape}")
    args.num_minerals += 64

# Define the list of elements
elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

if False: # FIX LATER 
    elements.remove('Silver')

elements.extend(['Fault', 'GeoAge Min', 'GeoAge Max', 'Elevation'])

if args.use_raca:
    elements.extend([f'RaCA_{i+1}' for i in range(64)])

actual_num_layers = counts.shape[1]
elements = elements[:actual_num_layers]

output_mineral = elements.index(args.output_mineral_name)
input_minerals = [i for i in range(len(elements)) if i != output_mineral]

# Train/test split
num_samples = len(counts)
num_test_samples = int(num_samples * 0.1)
test_indices = np.arange(num_test_samples)
train_indices = np.arange(num_test_samples, num_samples)

# Setup
train_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=train_indices, train=True, unet=(args.model_type == "u"))
test_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=test_indices, train=False, unet=(args.model_type == "u"))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Model selection
if args.model_type == "tc":
    model = TransformerToConv(input_dim=args.num_minerals * args.grid_size * args.grid_size, hidden_dim=args.hidden_dim, 
                              intermediate_dim=args.intermediate_dim, d_model=args.d_model, nhead=args.nhead, 
                              num_layers=args.num_layers, dropout_rate=args.dropout_rate)
elif args.model_type == "u":
    model = UNet(in_channels=args.num_minerals, out_channels=1)
elif args.model_type == "lc":
    model = LinToConv(input_dim=args.num_minerals * args.grid_size * args.grid_size, hidden_dim=args.hidden_dim, intermediate_dim=args.intermediate_dim)
elif args.model_type == "lt":
    model = LinToTransformer(input_dim=args.num_minerals * args.grid_size * args.grid_size, hidden_dim=args.hidden_dim, 
                             intermediate_dim=args.intermediate_dim, d_model=args.d_model, nhead=args.nhead, 
                             num_layers=args.num_layers, dropout_rate=args.dropout_rate)
elif args.model_type == "l":
    model = SimplifiedMLP(input_dim=args.num_minerals * args.grid_size * args.grid_size, hidden_dim=args.hidden_dim)
else:
    raise ValueError(f"Unknown model type: {args.model_type}")

model = model.to(device)

def train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, two_step=False, first_loss='integral', use_bce=False, include_true_negatives=False):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if use_bce:
        criterion = nn.BCEWithLogitsLoss()

    losses, test_losses, nonempty_losses, diff_integrals, dice_coefs = [], [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_nonempty_loss, total_diff_integral, total_dice_coef, num_batches = 0, 0, 0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nonempty_loss_value = nonempty_loss(outputs, targets)
            diff_integral = absolute_difference_integral(outputs, targets).item()
            dice_coef = dice_coefficient_nonzero(outputs, targets).item()

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

        print(f'Epoch {epoch+1}/{num_epochs} (Step 1), Train Loss: {avg_loss}, Test Loss: {test_loss}, Non-Empty Train Loss: {avg_nonempty_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Train): {avg_diff_integral}, Difference in Integrals (Test): {test_diff_integral}, Dice Coefficient (Train): {avg_dice_coef}, Dice Coefficient (Test): {avg_dice_coef}')

    if two_step:
        for epoch in range(num_epochs):
            model.train()
            total_loss, total_nonempty_loss, total_diff_integral, total_dice_coef, num_batches = 0, 0, 0, 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = combined_loss(outputs, targets, first_loss, include_true_negatives=include_true_negatives)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_nonempty_loss += nonempty_loss(outputs, targets).item()
                total_diff_integral += absolute_difference_integral(outputs, targets).item()
                total_dice_coef += dice_coefficient_nonzero(outputs, targets).item()
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

            print(f'Epoch {epoch+1}/{num_epochs} (Step 2), Train Loss: {avg_loss}, Test Loss: {test_loss}, Non-Empty Train Loss: {avg_nonempty_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Train): {avg_diff_integral}, Difference in Integrals (Test): {test_diff_integral}, Dice Coefficient (Train): {avg_dice_coef}, Dice Coefficient (Test): {avg_dice_coef}')

    final_criterion = lambda p, t: combined_loss(p, t, first_loss, include_true_negatives=include_true_negatives) if two_step else criterion(p, t)
    test_loss, nonempty_test_loss, predicted_output_test, output_tensor_test, final_diff_integral, final_dice_coef = evaluate(model, test_loader, final_criterion, use_bce=use_bce, include_true_negatives=include_true_negatives)
    print(f'Final Test Loss: {test_loss}, Non-Empty Test Loss: {nonempty_test_loss}, Difference in Integrals (Test): {final_diff_integral}, Dice Coefficient (Test): {final_dice_coef}')

    final_train_metrics = compute_metrics(train_loader, model, final_criterion, use_bce, include_true_negatives)
    final_test_metrics = compute_metrics(test_loader, model, final_criterion, use_bce, include_true_negatives)
    wandb.log(final_train_metrics)
    wandb.log(final_test_metrics)

    plot_training_curves(losses, test_losses, nonempty_losses, diff_integrals, dice_coefs)

    return predicted_output_test, output_tensor_test

def evaluate(model, data_loader, criterion, use_bce=False, include_true_negatives=False):
    model.eval()
    total_loss, total_nonempty_loss, total_diff_integral, total_dice_coef, num_batches = 0, 0, 0, 0, 0
    predicted_output, output_tensor = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets).item()
            nonempty_loss_value = nonempty_loss(outputs, targets).item()
            diff_integral = absolute_difference_integral(outputs, targets).item()
            dice_coef = dice_coefficient_nonzero(outputs, targets).item()

            total_loss += loss
            total_nonempty_loss += nonempty_loss_value
            total_diff_integral += diff_integral
            total_dice_coef += dice_coef
            num_batches += 1

            predicted_output.append(outputs)
            output_tensor.append(targets)

    avg_loss = total_loss / num_batches
    avg_nonempty_loss = total_nonempty_loss / num_batches
    avg_diff_integral = total_diff_integral / num_batches
    avg_dice_coef = total_dice_coef / num_batches

    predicted_output = torch.cat(predicted_output, dim=0)
    output_tensor = torch.cat(output_tensor, dim=0)

    return avg_loss, avg_nonempty_loss, predicted_output, output_tensor, avg_diff_integral, avg_dice_coef

def compute_metrics(data_loader, model, criterion, use_bce, include_true_negatives):
    model.eval()
    total_dice_coef, total_diff_integral, total_nonempty_mse, total_pixel_mse, num_batches = 0, 0, 0, 0, 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total_dice_coef += dice_coefficient_nonzero(outputs, targets).item()
            total_diff_integral += absolute_difference_integral(outputs, targets).item()
            total_nonempty_mse += nonempty_loss(outputs, targets).item()
            total_pixel_mse += regular_loss(outputs, targets).item()
            num_batches += 1

    return {
        "Avg Dice Coefficient": total_dice_coef / num_batches,
        "Avg Difference in Integral": total_diff_integral / num_batches,
        "Avg Non-Empty MSE": total_nonempty_mse / num_batches,
        "Avg Pixel MSE": total_pixel_mse / num_batches
    }

def plot_training_curves(losses, test_losses, nonempty_losses, diff_integrals, dice_coefs):
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

if __name__ == "__main__":
    predicted_output_test, output_tensor_test = train(
        model, train_loader, test_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        criterion=integral_loss if args.loss1 == "integral" else regular_loss, 
        two_step=args.two_step == 'True', 
        first_loss=args.loss1, 
        use_bce=args.use_bce, 
        include_true_negatives=args.tn
    )

    # Crop the 64x64 outputs back to 50x50
    crop_size = (50, 50)
    predicted_output_test = predicted_output_test[:, :, 0:crop_size[0], 0:crop_size[1]]
    output_tensor_test = output_tensor_test[:, :, 0:crop_size[0], 0:crop_size[1]]

    batch_size = predicted_output_test.shape[0]
    predicted_np_test = predicted_output_test.cpu().numpy().reshape(batch_size, 1, crop_size[0], crop_size[1])
    output_np_test = output_tensor_test.cpu().numpy().reshape(batch_size, 1, crop_size[0], crop_size[1])

    print(f"Predicted Test Shape: {predicted_np_test.shape}")
    print(f"Output Test Shape: {output_np_test.shape}")
    plot_predictions(predicted_np_test, output_np_test, counts[test_indices], input_minerals, fault_data[test_indices], geoAge_data[test_indices], elevations[test_indices], elements[output_mineral], num_samples=20, specs=args.logName)
