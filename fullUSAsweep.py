import torch
import h5py
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from models import UNet
from utils import dice_coefficient_nonzero, absolute_difference_integral, nonempty_loss, regular_loss

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GRID_SIZE = 50
PAD_SIZE = 64  # Adjust if needed
PADDING = (PAD_SIZE - GRID_SIZE) // 2
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded paths
USADATA_FILE_PATH = 'prepared_data_TILES/USADATA.h5'  
USADATA_DATASET = 'data'         # The dataset in USADATA.h5 with shape [N,15,50,50]
MODEL_PATH = "model_minerals_geophysical_no_raca.pth"  
MODEL_NAME = "minerals_geophysical_no_raca"

# Gold channel index
GOLD_CHANNEL = 0  # If gold is at channel 0

def load_hdf5_data(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

# Load the USADATA (GT)
gt_data = load_hdf5_data(USADATA_FILE_PATH, USADATA_DATASET)  # e.g. [N,15,50,50]
# Extract ground truth gold
gt_gold = gt_data[:, GOLD_CHANNEL:GOLD_CHANNEL+1, :, :]  # [N,1,50,50]

# Create masked input by zeroing out the gold channel
masked_input = gt_data.copy()
masked_input[:, GOLD_CHANNEL, :, :] = 0.0  # Mask out gold

# The model expects input_channels = masked_input.shape[1]
in_channels = masked_input.shape[1]

class USADataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = USADataset(masked_input, gt_gold)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def compute_metrics(predictions, ground_truth):
    # predictions, ground_truth: [B,C,H,W]
    mse = regular_loss(predictions, ground_truth).item()
    dice_coef = dice_coefficient_nonzero(predictions, ground_truth).item()
    diff_integral = absolute_difference_integral(predictions, ground_truth).item()
    nonempty_mse = nonempty_loss(predictions, ground_truth).item()
    return mse, dice_coef, diff_integral, nonempty_mse

def evaluate_model(model_path, model_name, in_channels):
    model = UNet(in_channels=in_channels, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    metrics = []
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for inputs, gt in data_loader:
            inputs = inputs.float()  # [B,in_channels,50,50]
            gt = gt.float()          # [B,1,50,50]

            # Pad inputs to 64x64 if needed by model
            inputs_padded = F.pad(inputs, (PADDING, PADDING, PADDING, PADDING))  # [B,in_channels,64,64]
            inputs_padded = inputs_padded.to(DEVICE)

            outputs_padded = model(inputs_padded)  # [B,1,64,64]
            # Crop back to 50x50
            outputs = outputs_padded[:, :, PADDING:PADDING+GRID_SIZE, PADDING:PADDING+GRID_SIZE].cpu()  # [B,1,50,50]

            predictions = outputs.numpy()   # [B,1,50,50]
            ground_truth = gt.numpy()        # [B,1,50,50]

            # Compute metrics per sample
            for i in range(len(predictions)):
                pred_tensor = torch.from_numpy(predictions[i]).unsqueeze(0) # [1,1,H,W]
                gt_tensor = torch.from_numpy(ground_truth[i]).unsqueeze(0)  # [1,1,H,W]
                m = compute_metrics(pred_tensor, gt_tensor)
                metrics.append(m)

            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)

    all_predictions = np.concatenate(all_predictions, axis=0)  # [N,1,50,50]
    all_ground_truth = np.concatenate(all_ground_truth, axis=0) # [N,1,50,50]

    # Save results
    np.save(os.path.join(OUTPUT_DIR, f"predictions_{model_name}.npy"), all_predictions)
    np.save(os.path.join(OUTPUT_DIR, f"ground_truth_{model_name}.npy"), all_ground_truth)

    avg_metrics = np.mean(metrics, axis=0)
    print(f"Model: {model_name}")
    print(f"Average MSE: {avg_metrics[0]:.3f}")
    print(f"Average Dice Coefficient: {avg_metrics[1]:.3f}")
    print(f"Average Difference in Integrals: {avg_metrics[2]:.3f}")
    print(f"Average Non-Empty MSE: {avg_metrics[3]:.3f}")

    # Plot a few comparisons
    for i in range(min(5, len(all_predictions))):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth (Gold)")
        plt.imshow(all_ground_truth[i, 0], cmap='viridis')
        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(all_predictions[i, 0], cmap='viridis')
        plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{model_name}_{i}.png"))
        plt.close()

if __name__ == "__main__":
    evaluate_model(MODEL_PATH, MODEL_NAME, in_channels)


import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to saved prediction numpy arrays
# These are the predictions you saved after running each model's inference script.
PREDICTIONS_MINERALS_ONLY = 'inference_results/predictions_minerals_only.npy'
PREDICTIONS_MINERALS_GEO = 'inference_results/predictions_minerals_geophysical_no_raca.npy'
GROUND_TRUTH = 'inference_results/ground_truth_minerals_geophysical_no_raca.npy'  # Same GT for both, just one saved set

# Load arrays
pred_minerals_only = np.load(PREDICTIONS_MINERALS_ONLY)   # [N,1,50,50]
pred_minerals_geo   = np.load(PREDICTIONS_MINERALS_GEO)   # [N,1,50,50]
gt_gold = np.load(GROUND_TRUTH)                            # [N,1,50,50]

# Compare the two predictions:
# For demonstration, we will average all tiles into one large aggregate if needed,
# or show just one tile as an example. For a full US map, you'd need to know how the 50x50 tiles
# map onto the US geography. Without that info, we just show a single tile's difference.

# Let's pick an arbitrary index (e.g., 0) to visualize:
tile_index = 0
pred_m_only_tile = pred_minerals_only[tile_index,0,:,:]  # [50,50]
pred_m_geo_tile  = pred_minerals_geo[tile_index,0,:,:]   # [50,50]
gt_tile          = gt_gold[tile_index,0,:,:]             # [50,50]

# Compute difference (geophysical - minerals_only)
difference_tile = pred_m_geo_tile - pred_m_only_tile

# Plot the differences as a heatmap
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Ground Truth Gold")
plt.imshow(gt_tile, cmap='viridis')
plt.colorbar(label='Gold concentration')

plt.subplot(1,3,2)
plt.title("Minerals-only Prediction")
plt.imshow(pred_m_only_tile, cmap='viridis')
plt.colorbar(label='Gold concentration')

plt.subplot(1,3,3)
plt.title("Geophysical - Minerals-only (Difference)")
# Positive values = geophysical model predicts more gold than minerals-only
# Negative values = geophysical model predicts less gold than minerals-only
plt.imshow(difference_tile, cmap='bwr', vmin=-abs(difference_tile).max(), vmax=abs(difference_tile).max())
plt.colorbar(label='Difference in Gold concentration')

plt.tight_layout()
plt.savefig('inference_results/model_comparison_heatmap.png')
plt.close()
