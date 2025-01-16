import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import random
import wandb
from torchmetrics.functional.classification import binary_precision_recall_curve
from torch.utils.data import Dataset, DataLoader
import os
from models import LinToConv, SimplifiedMLP, LinToTransformer, MineralDataset, UNet, TransformerToConv, SpatialTransformer
from utils import (integral_loss, regular_loss, nonempty_loss, 
                   dice_coefficient_nonzero, absolute_difference_integral, save_overlay_predictions, create_gif)

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for mineral prediction and save three models.")
parser.add_argument('--grid_size', type=int, default=50, help='Grid size of the input data.')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the models.')
parser.add_argument('--intermediate_dim', type=int, default=512, help='Intermediate dimension for additional layers.')
parser.add_argument('--num_minerals', type=int, default=15, help='Number of input dimensions (minerals).')
parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the multihead attention models.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the transformer.')
parser.add_argument('--d_model', type=int, default=256, help='Model dimension for the transformer.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the models.')
parser.add_argument('--model_type', type=str, default='u', choices=['tc', 'u', 'lc', 'lt', 'l', 'spatial_transformer'], help='Model type.')
parser.add_argument('--output_mineral_name', type=str, default='Gold', help='Name of the output mineral.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--loss1', type=str, default='integral', help='First loss function.')
parser.add_argument('--two_step', type=str, default='True', help='Whether to use two-step training.')
parser.add_argument('--logName', type=str, default='testB3', help='Name of the log file.')
parser.add_argument('--use_bce', action='store_true', help='Use binary cross-entropy loss for resource presence detection.')
parser.add_argument('--tn', action='store_true', help='Include true negatives in the evaluation.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
parser.add_argument('--set_seed', type=int, default=42, help='Set seed for reproducibility.')
parser.add_argument('--raca_odim', type=int, default=64, help='Set raca hidden dim')
parser.add_argument('--unetArch', type=str, default="resnet152", choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Unet encoder architecture')
#new flags:
parser.add_argument('--use_minerals', action='store_true', help='Include mineral data.')
parser.add_argument('--use_geophys', action='store_true', help='Include geophysical data.')
parser.add_argument('--use_raca_data', action='store_true', help='Include RaCA data (if also using --use_raca).')

args = parser.parse_args()

# Configuration
torch.manual_seed(args.set_seed)
np.random.seed(args.set_seed)
random.seed(args.set_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = vars(args)
# wandb.init(project="mineral_transformer_project", name=args.logName, config=config)

print("Device:", device)
print("Configuration:", config)

# HDF5 Paths
rocktype_file_path = '/home/sujaynair/MRDS_Project/GEOPHYSDATA_DEC/Rocktype.h5'
faults_file_path = '/home/sujaynair/MRDS_Project/GEOPHYSDATA_DEC/faults.h5'
geoage_file_path = '/home/sujaynair/MRDS_Project/GEOPHYSDATA_DEC/Geoage.h5'
elevations_file_path = '/home/sujaynair/MRDS_Project/GEOPHYSDATA_DEC/elevations.h5'
mineral_data_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
raca_file_path = '/home/sujaynair/MRDS_Project/newRaCA_DATA/raca_raw.pkl'

# Load datasets
def load_hdf5_data(file_path, dataset_name):
    print(f"Loading dataset {dataset_name} from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
        print(f"Loaded {dataset_name} with shape {data.shape}")
        return data

def normalize_layer(layer):
    min_val, max_val = np.min(layer), np.max(layer)
    normalized = (layer - min_val) / (max_val - min_val) if max_val != min_val else layer
    print(f"Normalized layer with min {min_val} and max {max_val}")
    return normalized

if args.use_minerals:
    print("Loading mineral data...")
    mineral_data = load_hdf5_data(mineral_data_path, 'counts')
    mineral_data = np.nan_to_num(mineral_data, nan=-10)
    print("Mineral data shape:", mineral_data.shape)
else:
    mineral_data = None

if args.use_geophys:
    print("Loading geophysical data...")
    rock_type = normalize_layer(load_hdf5_data(rocktype_file_path, 'rock_type'))
    fault_presence = normalize_layer(load_hdf5_data(faults_file_path, 'fault_presence'))
    fault_slip_rate = normalize_layer(load_hdf5_data(faults_file_path, 'fault_slip_rate'))
    maximum_age = normalize_layer(load_hdf5_data(geoage_file_path, 'maximum_age'))
    minimum_age = normalize_layer(load_hdf5_data(geoage_file_path, 'minimum_age'))
    elevations = normalize_layer(load_hdf5_data(elevations_file_path, 'elevations'))
    geophysical_data = np.stack([rock_type, fault_presence, fault_slip_rate, maximum_age, minimum_age, elevations], axis=1)
    print("Geophysical data shape:", geophysical_data.shape)
else:
    geophysical_data = None

if args.use_raca_data:
    print("Loading RaCA data...")
    with open(raca_file_path, 'rb') as file:
        raca_data = pickle.load(file)
    print("RaCA data loaded successfully.")
else:
    raca_data = None

# Combine datasets dynamically
combined_data = []
if args.use_minerals:
    combined_data.append(mineral_data)
if args.use_geophys:
    combined_data.append(geophysical_data)

if combined_data:
    combined_data = np.concatenate(combined_data, axis=1)
    print("Combined minerals and geophysical data shape:", combined_data.shape)
else:
    combined_data = None
    print("No data selected for training.")
# # Load RaCA data
# raca_data = load_hdf5_data(raca_file_path, 'aggregated_output')
# raca_data = raca_data.transpose((0, 3, 1, 2))  # Transpose to (N, C, H, W)
# raca_data = np.nan_to_num(raca_data, nan=-10)
# for i in range(raca_data.shape[1]):
#     raca_data[:, i, :, :] = normalize_layer(raca_data[:, i, :, :])
# print("RaCA data shape:", raca_data.shape)

# # Combined Minerals, Geophysical, and RaCA
# combined_geo_raca_data = np.concatenate((combined_data, raca_data), axis=1)
# print("Combined minerals, geophysical, and RaCA data shape:", combined_geo_raca_data.shape)

# Metadata
elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese'] + \
           ['Rock Type', 'Fault Presence', 'Fault Slip Rate', 'Max Age', 'Min Age', 'Elevation']
# elements += [f'RaCA_{i + 1}' for i in range(raca_data.shape[1])]

# Data Splits
num_samples = 10000
num_test_samples = int(num_samples * 0.1)
train_indices = np.arange(num_test_samples, num_samples)
test_indices = np.arange(num_test_samples)
print("Number of samples:", num_samples)
print("Number of training samples:", len(train_indices))
print("Number of test samples:", len(test_indices))

def evaluate(model, data_loader, criterion, raca_data=None, raca_encoder=None):
    model.eval()
    if args.use_raca_data:
        assert raca_data is not None and raca_encoder is not None, "RaCA data and encoder must be provided for evaluation if --use_raca_data is enabled."
        raca_encoder.eval()

    total_loss, total_batches = 0, 0
    total_dice_coef = []

    with torch.no_grad():
        for inputs, targets, idx, mask in data_loader: #FIX FOR MASK
            inputs, targets = inputs.to(device), targets.to(device)

            # Conditionally add RaCA embeddings if enabled
            if args.use_raca_data:
                raca_embeddings = getRACAembed(raca_data, idx, raca_encoder)
                inputs = torch.concatenate((inputs, raca_embeddings), dim=1)

            # Forward pass
            outputs = model(inputs)

            # Accumulate Loss
            total_loss += criterion(outputs, targets).item()

            # Accumulate Dice Coefficient
            total_dice_coef.append(dice_coefficient_nonzero(outputs, targets).item())

            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_dice = torch.mean(torch.Tensor(total_dice_coef)) if total_batches > 0 else 0

    print(f"Evaluation - Avg Loss: {avg_loss:.4f}, Avg Dice Coefficient: {avg_dice:.4f}")
    return avg_loss, avg_dice

def create_model(args, num_channels):
    print(f"Creating model of type {args.model_type} with {num_channels} input channels...")
    if args.model_type == "tc":
        model = TransformerToConv(
            input_dim=num_channels * args.grid_size * args.grid_size,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
    elif args.model_type == "spatial_transformer":
        model = SpatialTransformer(
            in_channels = num_channels,
            out_channels=10 if args.output_mineral_name == "all" else 1,
            hidden_dim=32,
            num_layers=2
            )
    elif args.model_type == "u":
        model = UNet(in_channels=num_channels, out_channels=10 if args.output_mineral_name == "all" else 1, arch=args.unetArch)
    elif args.model_type == "lc":
        model = LinToConv(
            input_dim=num_channels * args.grid_size * args.grid_size,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim
        )
    elif args.model_type == "lt":
        model = LinToTransformer(
            input_dim=num_channels * args.grid_size * args.grid_size,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
    elif args.model_type == "l":
        model = SimplifiedMLP(
            input_dim=num_channels * args.grid_size * args.grid_size,
            hidden_dim=args.hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    print("Model created successfully.")
    return model.to(device)

def getRACAembed(raca, indices, encoder):
    indices = indices.squeeze().tolist()
    output = torch.zeros((len(indices), 50, 50, args.raca_odim)).to(device)
    for idx_, idx in enumerate(indices):
        for lat_idx, lat in enumerate(raca[idx]):
            for lon_idx, lon in enumerate(lat):
                embeddings = []
                for scan_idx, scan in enumerate(lon):
                    scan = torch.FloatTensor(scan).unsqueeze(0).to(device)
                    embedding = encoder(scan)
                    embeddings.append(embedding)
                if len(embeddings) > 0:
                    embeddings = torch.stack(embeddings)
                    output[idx_, lat_idx, lon_idx] = embeddings.mean(0)

    permuted_output = output.permute(0, 3, 1, 2)
    if args.model_type == "u":
        pad = (0, 14, 0, 14)
        permuted_output = F.pad(permuted_output, pad, mode='constant', value=0)

    return permuted_output



class Conv1DEncoder(nn.Module):
    def __init__(self, input_dim=2150, output_dim=args.raca_odim, hidden_channels=64):
        super(Conv1DEncoder, self).__init__()
        self.input_dim = input_dim

        # 1D convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, stride=2, padding=2)

        # Fully connected layer to produce the final embedding
        self.fc = nn.Linear(hidden_channels * (input_dim // 8), output_dim)  # Update based on downsampling

    def forward(self, x):
        # Input: (batch_size, input_dim)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_dim)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)  # (batch_size, hidden_channels * reduced_dim)

        # Final embedding
        embedding = self.fc(x)  # (batch_size, output_dim)
        return embedding
    
def compute_aux_metrics(outputs, target, mask, prefix = "train"):
    metrics = {}

    # print(outputs.shape, target.shape)
    outputs_agg, _ = outputs.max(dim=1)
    target_agg, _ = target.max(dim=1)
    mask_agg, _ = mask.max(dim=1)

    # Masked outputs and targets
    outputs_masked = torch.masked_select(outputs, mask.bool())
    targets_masked = torch.masked_select(target, mask.bool())

    outputs_agg_masked = torch.masked_select(outputs_agg, mask_agg.bool())
    target_agg_masked = torch.masked_select(target_agg, mask_agg.bool())

    # DICE
    metrics[f"{prefix}/DICE_Avg"] = dice_coefficient_nonzero(outputs_masked, targets_masked)
    metrics[f"{prefix}/AGG_DICE_Avg"] = dice_coefficient_nonzero(outputs_agg_masked, target_agg_masked)

    # Integral MSE
    integral_mask = torch.amax(mask, dim = (2,3))
    intergral_mse_unreduced = F.mse_loss(torch.sum((outputs * mask), dim=[2, 3]), torch.sum((target * mask), dim=[2, 3]), reduction = "none")
    metrics[f"{prefix}/IntegralMSE_Avg"] = torch.masked_select(intergral_mse_unreduced, integral_mask.bool()).mean()

    # Pixel MSE
    metrics[f"{prefix}/PixelMSE_Avg"] = F.mse_loss(outputs_masked, targets_masked, reduction='mean')

    # Precision and Recall
    precision, recall, thresholds = binary_precision_recall_curve(outputs_masked, targets_masked.int(), thresholds = 20, validate_args=True)
    # print(outputs_masked)
    # print(targets_masked)
    # print(precision, recall, thresholds)
    # for idx, threshold in enumerate(thresholds):
    #     metrics[f"{prefix}/Precision_{threshold}_Avg"] = precision[idx].mean()
    #     metrics[f"{prefix}/Recall_{threshold}_Avg"] = recall[idx].mean()

    return metrics, precision, recall



    

def prepare_and_train(data, suffix):
    print(f"Preparing and training model: {suffix}...")
    actual_num_layers = 0
    selected_data = []
    used_flags = []  # Track which flags are used for dynamic naming

    if args.use_minerals:
        actual_num_layers += mineral_data.shape[1]
        selected_data.append(mineral_data)
        used_flags.append("minerals")
        print("Including mineral data...")

    if args.use_geophys:
        actual_num_layers += geophysical_data.shape[1]
        selected_data.append(geophysical_data)
        used_flags.append("geophys")
        print("Including geophysical data...")

    if args.use_raca_data:
        actual_num_layers += args.raca_odim  # Assuming RACA embeddings add 32 channels
        used_flags.append("raca")
        print("Including RaCA data...")

    # Combine selected datasets
    if selected_data:
        data = np.concatenate(selected_data, axis=1)
        print("Combined data shape after selection:", data.shape)
    else:
        raise ValueError("No data selected for training. Use at least one of --use_minerals, --use_geophys, or --use_raca_data.")

    # Dynamically set WandB log name based on flags
    log_name = f"{args.logName}_{'_'.join(used_flags)}"
    wandb.init(project="mineral_transformer_project", name=log_name, config=vars(args))

    if args.output_mineral_name == "all":
        output_mineral = None
    else:
        output_mineral = elements.index(args.output_mineral_name)
    input_minerals = [i for i in range(len(elements)) if i != output_mineral]

    train_dataset = MineralDataset(
        data,
        input_minerals,
        output_mineral,
        indices=train_indices,
        train=True,
        unet=(args.model_type == "u"),
    )
    test_dataset = MineralDataset(
        data,
        input_minerals,
        output_mineral,
        indices=test_indices,
        train=False,
        unet=(args.model_type == "u"),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = create_model(args, actual_num_layers).to(device)
    raca_encoder = Conv1DEncoder(input_dim=2136).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.AdamW(
        list(model.parameters()) + list(raca_encoder.parameters()), lr=args.learning_rate
    )

    # ------------------
    #     Training
    # ------------------
    for epoch in range(args.num_epochs):
        model.train()
        raca_encoder.train()
        total_train_loss = 0
        train_metrics = {}
        train_precisions = []
        train_recalls = []
        for inputs, targets, idx, mask in train_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()

            if args.use_raca_data:
                raca_embeddings = getRACAembed(raca_data, idx, raca_encoder)
                inputs = torch.concatenate((inputs, raca_embeddings), dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            with torch.no_grad():
                aux_metrics, train_precision, train_recall = compute_aux_metrics(outputs, targets, mask)
                train_precisions.append(train_precision)
                train_recalls.append(train_recall)
                if train_metrics:
                    train_metrics = {k : v + aux_metrics[k] for k,v in train_metrics.items()}
                else:
                    train_metrics = aux_metrics

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        train_precisions = torch.stack(train_precisions).mean(0).cpu().detach().numpy()
        train_recalls = torch.stack(train_recalls).mean(0).cpu().detach().numpy()
        plt.plot(train_recalls, train_precisions)
        plt.title("PR_curve")
        plt.xlabel("Train_Recall")
        plt.ylabel("Train_Precision")
        plt.savefig("train_pr_curve.png")
        plt.close()
        with torch.no_grad():
            all_train_images = {}
            if args.output_mineral_name == "all":
                plot_indices = list(range(10))
            else:
                plot_indices = [0]
            for k in plot_indices:
                all_train_images[k] = save_overlay_predictions(outputs, targets, mask, k, f'/home/sujaynair/MRDS_Project/plotOutputsDec/train_{epoch}.png')

        # ------------------
        #     Testing
        # ------------------
        model.eval()
        raca_encoder.eval()
        total_test_loss = 0
        test_metrics = {}
        test_precisions = []
        test_recalls = []
        with torch.no_grad():
            for inputs, targets, idx, mask in test_loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

                if args.use_raca_data:
                    raca_embeddings = getRACAembed(raca_data, idx, raca_encoder)
                    inputs = torch.concatenate((inputs, raca_embeddings), dim=1)

                outputs = model(inputs)
                total_test_loss += criterion(outputs, targets).item()

                aux_metrics, test_precision, test_recall = compute_aux_metrics(outputs, targets, mask, prefix = "test")
                test_precisions.append(test_precision)
                test_recalls.append(test_recall)
                if test_metrics:
                    test_metrics = {k : v + aux_metrics[k] for k,v in test_metrics.items()}
                else:
                    test_metrics = aux_metrics

            all_test_images = {}
            for k in plot_indices:
                all_test_images[k] = save_overlay_predictions(outputs, targets, mask, k, f'/home/sujaynair/MRDS_Project/plotOutputsDec/test_{epoch}.png')

        avg_test_loss = total_test_loss / len(test_loader)
        test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}

        test_precisions = torch.stack(test_precisions).mean(0).cpu().detach().numpy()
        test_recalls = torch.stack(test_recalls).mean(0).cpu().detach().numpy()
        plt.plot(test_recalls, test_precisions)
        plt.title("PR_curve")
        plt.xlabel("Test_Recall")
        plt.ylabel("Test_Precision")
        plt.savefig("test_pr_curve.png")
        plt.close()

        # Log metrics to WandB
        logs = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
        } | train_metrics | test_metrics
        for k in plot_indices:
            logs[f"Train viz at mineral {k}"] = wandb.Image(all_train_images[k])
            logs[f"Test viz at mineral {k}"] = wandb.Image(all_test_images[k])

        logs["train/pr_curve"] = wandb.Image("train_pr_curve.png")
        logs["test/pr_curve"] = wandb.Image("test_pr_curve.png")

        wandb.log(logs)
        print("-" * 50)
        print("-" * 50)
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}")
        for k, v in logs.items():
            if "DICE" in k or "MSE" in k:
              print(f"Aux Metric {k}: {v}")
        print("-" * 50)
        print("-" * 50)
    # ------------------
    #   Save the Model
    # ------------------
        if epoch == 250:
            torch.save({
                "model_state_dict": model.state_dict(),
                "raca_encoder_state_dict": raca_encoder.state_dict(),
            }, f"model_{suffix}_{log_name}.pth")
            print(f"Model {suffix} saved successfully at epoch: {epoch}")

    # ------------------
    #     Evaluate
    # ------------------
    # avg_loss, avg_dice = evaluate(model, test_loader, criterion, raca_data, raca_encoder)

    # Log evaluation metrics to WandB
    # wandb.log({
    #     "avg_loss": avg_loss,
    #     "avg_dice_coefficient": avg_dice,
    # })

    # print(f"Model {suffix} evaluation completed: "
    #       f"Avg Loss: {avg_loss:.4f}, Avg Dice Coefficient: {avg_dice:.4f}")

    # End WandB logging
    wandb.finish()

# Call prepare_and_train with combined data
prepare_and_train(combined_data, "0115")
# Paths
image_folder = "/home/sujaynair/MRDS_Project/plotOutputsDec"  # Replace with the path to your folder
train_gif_name = "train.gif"
test_gif_name = "test.gif"

# Create GIFs
create_gif(image_folder, train_gif_name, "train_")
create_gif(image_folder, test_gif_name, "test_")

# prepare_and_train(mineral_data, "minerals_only")
# prepare_and_train(combined_data, "minerals_geophysical")
# prepare_and_train(combined_geo_raca_data, "minerals_geophysical_raca")
