import torch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import UNet
from utils import dice_coefficient_nonzero, absolute_difference_integral, nonempty_loss, regular_loss

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GRID_SIZE = 50
PAD_SIZE = 64  
PADDING = (PAD_SIZE - GRID_SIZE) // 2
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths to models
MODEL_PATH_MINERALS_ONLY = "model_minerals_only.pth"
MODEL_PATH_MINERALS_GEO = "model_minerals_geophysical_no_raca.pth"

MODEL_NAME_MINERALS_ONLY = "minerals_only"
MODEL_NAME_MINERALS_GEO = "minerals_geophysical_no_raca"

# Assume these paths and dataset are the same as in training
USADATA_FILE_PATH = 'prepared_data_TILES/USADATA.h5'  
USADATA_DATASET = 'data'

GOLD_CHANNEL = 0
MINERALS_ONLY_CHANNELS = 10
MINERALS_GEO_CHANNELS = 15

import h5py

def load_hdf5_data(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

# Load ground truth data
gt_data = load_hdf5_data(USADATA_FILE_PATH, USADATA_DATASET)  # shape [N,15,50,50]
gt_gold = gt_data[:, GOLD_CHANNEL:GOLD_CHANNEL+1, :, :]

# Create masked input by zeroing out gold channel
masked_input = gt_data.copy()
masked_input[:, GOLD_CHANNEL, :, :] = 0.0

class USADataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 1) Minerals-only dataset
masked_input_monly = masked_input[:, :MINERALS_ONLY_CHANNELS, :, :]
dataset_monly = USADataset(masked_input_monly, gt_gold)
data_loader_monly = DataLoader(dataset_monly, batch_size=BATCH_SIZE, shuffle=False)

# 2) Minerals+geo dataset
masked_input_geo = masked_input[:, :MINERALS_GEO_CHANNELS, :, :]
dataset_geo = USADataset(masked_input_geo, gt_gold)
data_loader_geo = DataLoader(dataset_geo, batch_size=BATCH_SIZE, shuffle=False)

def compute_metrics(predictions, ground_truth):
    mse = regular_loss(predictions, ground_truth).item()
    dice_coef = dice_coefficient_nonzero(predictions, ground_truth).item()
    diff_integral = absolute_difference_integral(predictions, ground_truth).item()
    nonempty_mse = nonempty_loss(predictions, ground_truth).item()
    return mse, dice_coef, diff_integral, nonempty_mse

def evaluate_model(model_path, model_name, in_channels, data_loader):
    model = UNet(in_channels=in_channels, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    metrics = []

    with torch.no_grad():
        for inputs, gt in data_loader:
            inputs = inputs.float()
            gt = gt.float()

            inputs_padded = F.pad(inputs, (PADDING, PADDING, PADDING, PADDING)).to(DEVICE)
            outputs_padded = model(inputs_padded)
            outputs = outputs_padded[:, :, PADDING:PADDING+GRID_SIZE, PADDING:PADDING+GRID_SIZE].cpu()

            for i in range(outputs.shape[0]):
                pred_tensor = outputs[i].unsqueeze(0)
                gt_tensor = gt[i].unsqueeze(0)
                m = compute_metrics(pred_tensor, gt_tensor)
                metrics.append(m)

    avg_metrics = np.mean(metrics, axis=0)
    print(f"Model: {model_name}")
    print(f"Average MSE: {avg_metrics[0]:.3f}")
    print(f"Average Dice Coefficient: {avg_metrics[1]:.3f}")
    print(f"Average Difference in Integral: {avg_metrics[2]:.3f}")
    print(f"Average Non-Empty MSE: {avg_metrics[3]:.3f}\n")
    return avg_metrics

if __name__ == "__main__":
    # Evaluate Minerals-Only Model
    print("Evaluating Minerals-Only Model:")
    avg_monly = evaluate_model(MODEL_PATH_MINERALS_ONLY, MODEL_NAME_MINERALS_ONLY, MINERALS_ONLY_CHANNELS, data_loader_monly)

    # Evaluate Minerals+Geophysical Model
    print("Evaluating Minerals+Geophysical Model:")
    avg_mgeo = evaluate_model(MODEL_PATH_MINERALS_GEO, MODEL_NAME_MINERALS_GEO, MINERALS_GEO_CHANNELS, data_loader_geo)

    # Evaluate on all USA squares
    usa_metrics = {
        MODEL_NAME_MINERALS_ONLY: avg_monly,
        MODEL_NAME_MINERALS_GEO: avg_mgeo
    }

    # Print a final comparison table
    print("Final Comparison Across All USA Squares:")
    print("                 MSE    Dice    Integral Diff    Non-Empty MSE")
    for model_name, metrics in usa_metrics.items():
        print(f"{model_name}: {metrics[0]:.3f}  {metrics[1]:.3f}      {metrics[2]:.3f}          {metrics[3]:.3f}")
