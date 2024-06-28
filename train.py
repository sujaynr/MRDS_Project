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

from models import LinToConv, SimplifiedMLP, LinToTransformer, MineralDataset
from utils import plot_predictions, integral_loss, evaluate, train

grid_size = 50  # Adjusted to 50x50 grid
hidden_dim = 512  # Adjust as necessary
intermediate_dim = 1024  # New intermediate dimension for additional layers
num_minerals = 10
nhead = 5  # Number of heads in the multiheadattention models
num_layers = 1  # Number of layers in the transformer
d_model = 500
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
non_empty_indices = np.sort(non_empty_indices)  # Sort indices

# Split into training and testing sets (20% for testing)
num_samples = len(non_empty_indices)
num_test_samples = int(num_samples * 0.1)
test_indices = non_empty_indices[:num_test_samples]
train_indices = non_empty_indices[:]  # Use the rest for training, OR ALL LIKE RN

print(f"Shape of dataset after removing empty squares: {counts[non_empty_indices].shape}")

# Layer mapping
elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
input_minerals = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # Indices for elements excluding Nickel
output_mineral = 5  # Index for Nickel

print("Layer mapping to elements:")
for i, element in enumerate(elements):
    print(f"Layer {i}: {element}")


# Create Dataset and DataLoader
train_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=train_indices, train=True)
test_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=test_indices, train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model and train
input_dim = num_minerals * grid_size * grid_size
# model = SimplifiedMLP(input_dim, hidden_dim)
model = LinToConv(input_dim, hidden_dim, intermediate_dim)
# model = LinToTransformer(input_dim, hidden_dim, intermediate_dim, d_model, nhead, num_layers, dropout_rate)
model = model.to(device)
pdb.set_trace()
predicted_output_test, output_tensor_test = train(model, train_loader, test_loader, num_epochs=30, learning_rate=0.00001)

# After the training process, reshape tensors back to the original shape for visualization and metric computation
batch_size = predicted_output_test.shape[0]
predicted_np_test = predicted_output_test.cpu().numpy().reshape(batch_size, 1, grid_size, grid_size)
output_np_test = output_tensor_test.cpu().numpy()

print(f"Predicted Test Shape: {predicted_np_test.shape}")
print(f"Output Test Shape: {output_np_test.shape}")


# Plot the predictions
plot_predictions(predicted_np_test, output_np_test, num_samples=20)

print("Prediction visualizations saved in the 'predictionVis' folder.")






























































# import wandb
# import os
# import h5py
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# from models import SimplifiedMineralTransformer

# # Initialize wandb
# wandb.init(project="mineral_transformer_project", name="simplified_model_test")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# grid_size = 50
# d_model = 64
# num_minerals = 10  # ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

# # Load data from HDF5 file
# h5_file_path = 'prepared_data_TILES/mineral_data.h5'

# # Load counts data to find non-empty squares
# with h5py.File(h5_file_path, 'r') as f:
#     counts = f['counts'][:]

# # Find non-empty squares and specifically ensure the Nickel layer is non-empty
# nickel_non_empty_indices = np.where(np.sum(counts[:, 5, :, :], axis=(1, 2)) > 0)[0]
# non_empty_indices = np.intersect1d(nickel_non_empty_indices, np.where(np.sum(counts, axis=(1, 2, 3)) > 50)[0])
# np.random.shuffle(non_empty_indices)
# non_empty_indices = np.sort(non_empty_indices)  # Sort indices

# print("Original shape of dataset:", counts.shape)
# print("Shape of dataset after removing empty squares:", counts[non_empty_indices].shape)

# # Use the same data for both training and testing to create a baseline
# train_indices = non_empty_indices
# test_indices = non_empty_indices

# class MineralDataset(Dataset):
#     def __init__(self, counts, input_minerals, output_mineral, indices, train=True, sigma=1):
#         self.counts = counts[indices]  # Load only non-empty squares based on indices
#         self.input_minerals = input_minerals  # Indices of input minerals
#         self.output_mineral = output_mineral  # Index of output mineral
#         self.train = train  # Flag for training mode
#         self.sigma = sigma  # Sigma for Gaussian filter

#     def __len__(self):
#         return len(self.counts)  # Return number of samples

#     def __getitem__(self, idx):
#         input_data = self.counts[idx].copy()  # Copy input data
#         output_data = input_data[self.output_mineral:self.output_mineral+1, :, :]  # Extract output data

#         # Apply Gaussian filter for smearing input and output data
#         input_data = gaussian_filter(input_data, sigma=self.sigma, mode='constant', truncate=3.0)
#         output_data = gaussian_filter(output_data, sigma=self.sigma, mode='constant', truncate=3.0)
        
#         # Normalize to keep total count the same, avoiding division by zero
#         epsilon = 1e-8  # Small value to avoid division by zero
#         input_sum = input_data.sum(axis=(1, 2), keepdims=True)
#         output_sum = output_data.sum(axis=(1, 2), keepdims=True)
        
#         if input_sum.sum() > 0:
#             input_data = np.where(input_sum > 0, input_data / (input_sum + epsilon) * self.counts[idx].sum(axis=(1, 2), keepdims=True), input_data)
#         if output_sum.sum() > 0:
#             output_data = np.where(output_sum > 0, output_data / (output_sum + epsilon) * self.counts[idx, self.output_mineral:self.output_mineral+1, :, :].sum(axis=(1, 2), keepdims=True), output_data)

#         # Convert data to PyTorch tensors
#         return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)

# # Custom loss function to calculate integral loss only for Nickel layer
# def custom_loss(predicted, target):
#     predicted_sum = torch.sum(predicted[:, 0, :, :], dim=[1, 2])  # Sum over the grid for Nickel layer
#     target_sum = torch.sum(target[:, 0, :, :], dim=[1, 2])  # Sum over the grid for Nickel layer
#     loss = nn.MSELoss()(predicted_sum, target_sum)  # MSELoss on the sums
#     return loss

# # Function to evaluate the model on a given dataset
# def evaluate(model, data_loader):
#     model.eval()  # Set model to evaluation mode
#     total_loss = 0  # Initialize total loss
#     predicted_output = []  # List to store predicted outputs
#     output_tensor = []  # List to store ground truth outputs

#     with torch.no_grad():  # No gradient computation
#         for input_tensor, output in data_loader:
#             input_tensor = input_tensor.to(device)  # Move input to device
#             output = output.to(device)  # Move output to device
#             outputs = model(input_tensor, output)  # Get model predictions

#             loss = custom_loss(outputs, output)  # Calculate loss
#             total_loss += loss.item()  # Accumulate loss

#             predicted_output.append(outputs)  # Store predictions
#             output_tensor.append(output)  # Store ground truth

#     avg_loss = total_loss / len(data_loader)  # Average loss
#     predicted_output = torch.cat(predicted_output, dim=0)  # Concatenate predictions
#     output_tensor = torch.cat(output_tensor, dim=0)  # Concatenate ground truth

#     return avg_loss, predicted_output, output_tensor

# def train(model, train_loader, test_loader, num_epochs=20, learning_rate=0.00001):
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer
#     criterion = custom_loss  # Custom loss function
#     losses = []  # List to store training losses
#     test_losses = []  # List to store test losses

#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode
#         total_loss = 0  # Initialize total loss
        
#         for input_tensor_train, output_tensor_train in train_loader:
#             # Apply Gaussian filter for smearing input and output data
#             input_tensor_train = torch.tensor(gaussian_filter(input_tensor_train.cpu().numpy(), sigma=1, mode='constant', truncate=3.0), dtype=torch.float32).to(device)
#             output_tensor_train = torch.tensor(gaussian_filter(output_tensor_train.cpu().numpy(), sigma=1, mode='constant', truncate=3.0), dtype=torch.float32).to(device)

#             optimizer.zero_grad()  # Zero gradients
#             outputs = model(input_tensor_train, output_tensor_train)  # Get model predictions

#             loss = criterion(outputs, output_tensor_train)  # Calculate loss
#             loss.backward()  # Backpropagate loss

#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
#             optimizer.step()  # Update weights

#             total_loss += loss.item()  # Accumulate loss

#         avg_loss = total_loss / len(train_loader)  # Average loss
#         losses.append(avg_loss)  # Store average loss

#         test_loss, _, _ = evaluate(model, test_loader)  # Evaluate model on test set
#         test_losses.append(test_loss)  # Store test loss

#         wandb.log({"Train Loss": avg_loss, "Test Loss": test_loss})  # Log losses to wandb

#         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss}, Test Loss: {test_loss}')  # Print losses

#     # Evaluate final model on test set
#     test_loss, predicted_output_test, output_tensor_test = evaluate(model, test_loader)

#     print(f'Final Test Loss: {test_loss}')  # Print final test loss

#     # Plot training and test loss curves
#     plt.plot(losses, label='Train Loss')
#     plt.plot(test_losses, label='Test Loss')
#     plt.title('Loss Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('trainingVis/loss_curve.png')
#     plt.close()
    
#     wandb.save('trainingVis/loss_curve.png')  # Save loss curve plot to wandb

#     return predicted_output_test, output_tensor_test  # Return final predictions and ground truth

# # Select input minerals (including Nickel) and output mineral (Nickel)
# input_minerals = list(range(num_minerals))  # Indices for all minerals
# output_mineral = 5  # Index for Nickel

# # Create Dataset and DataLoader for training and testing
# train_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=train_indices, train=True)
# test_dataset = MineralDataset(counts, input_minerals, output_mineral, indices=test_indices, train=False)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)  # Batch size of 10 for testing

# # Initialize model and train
# model = SimplifiedMineralTransformer(d_model=d_model)
# model = model.to(device)
# predicted_output_test, output_tensor_test = train(model, train_loader, test_loader, num_epochs=50, learning_rate=0.0001)

# # Reshape tensors back to the original shape for visualization and metric computation
# batch_size = predicted_output_test.shape[0]
# predicted_np_test = predicted_output_test.cpu().numpy().reshape(batch_size, 1, grid_size, grid_size)
# output_np_test = output_tensor_test.cpu().numpy()

# # Enhanced Visualization
# def visualize_predictions(predicted, ground_truth, idx, phase):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(predicted[idx, 0, :, :], cmap='viridis')
#     axes[0].set_title(f'Predicted Nickel ({phase})')
#     axes[1].imshow(ground_truth[idx, 0, :, :], cmap='viridis')
#     axes[1].set_title(f'Ground Truth Nickel ({phase})')
#     plt.show()

# # Visualize the 10 test samples
# if not os.path.exists("tilingVIS"):
#     os.makedirs("tilingVIS")

# for i in range(10):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(predicted_np_test[i, 0, :, :], cmap='viridis')
#     axes[0].set_title('Predicted Nickel (Test)')
#     axes[1].imshow(output_np_test[i, 0, :, :], cmap='viridis')
#     axes[1].set_title('Ground Truth Nickel (Test)')
#     plt.savefig(f"tilingVIS/test_sample_{i}.png")
#     plt.close()

# print("Test visualizations saved in the 'tilingVIS' folder.")

# # Visualize some training samples
# train_samples_to_visualize = min(10, len(train_loader.dataset))

# for i in range(train_samples_to_visualize):
#     input_data, output_data = train_loader.dataset[i]
#     input_data = input_data.unsqueeze(0).to(device)
#     output_data = output_data.unsqueeze(0).to(device)

#     with torch.no_grad():
#         predicted_output = model(input_data, output_data)

#     predicted_np_train = predicted_output.cpu().numpy().reshape(1, 1, grid_size, grid_size)
#     output_np_train = output_data.cpu().numpy().reshape(1, 1, grid_size, grid_size)

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(predicted_np_train[0, 0, :, :], cmap='viridis')
#     axes[0].set_title('Predicted Nickel (Train)')
#     axes[1].imshow(output_np_train[0, 0, :, :], cmap='viridis')
#     axes[1].set_title('Ground Truth Nickel (Train)')
#     plt.savefig(f"tilingVIS/train_sample_{i}.png")
#     plt.close()

# print("Train visualizations saved in the 'tilingVIS' folder.")

# # Finish the wandb run
# wandb.finish()




''' Memorization script
import wandb
import os
import pickle
import pdb
import numpy as np
import torch
from utils import visualize_layers, compute_dice_coefficients, plot_dice_coefficients, plot_metric, weighted_mse_loss
from models import MineralTransformer


import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Initialize wandb
wandb.init(project="mineral_transformer_project", name="2048_0.0001TEST")

grid_size = 60
d_model = 256

losses = []

def train(model, input_tensor_train, output_tensor_train, num_epochs=500, learning_rate=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    batch_size, seq_length_input, feature_dim = input_tensor_train.shape
    seq_length_output = output_tensor_train.shape[1]

    # We can penalize quality layers differently with this
    weights = torch.ones_like(output_tensor_train)
    weights[:, :5, :] = 1  # Set weights for A resources (for now not weighting with 1)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputsCS, outputs = model(input_tensor_train, output_tensor_train)

        # Flatten for loss 
        flat_outputsCS = outputsCS.view(batch_size, -1)
        flat_output_tensor_train = output_tensor_train.view(batch_size, -1)

        lossCS = weighted_mse_loss(flat_outputsCS, flat_output_tensor_train, weights.view(batch_size, -1))
        lossCS.backward()
        
        flat_outputs = outputs.view(batch_size, -1)


        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(lossCS.item())
        wandb.log({"Loss": lossCS.item()})

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {lossCS.item()}')

    plt.plot(losses, label='Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('trainingVis/loss_curve.png')
    plt.close()

    wandb.save('trainingVis/loss_curve.png')

data_dir = 'prepared_data_60x60'
elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Lead', 'Copper']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)

for elem in elements:
    print(f"{elem} - A layer sum: {np.sum(data[elem][0])}")
    print(f"{elem} - B layer sum: {np.sum(data[elem][1])}")
    print(f"{elem} - C layer sum: {np.sum(data[elem][2])}")
    print(f"{elem} - D layer sum: {np.sum(data[elem][3])}")
    print(f"{elem} - E layer sum: {np.sum(data[elem][4])}")

output_elements = ['Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

output_layers = np.stack([data[elem] for elem in output_elements], axis=0)
# Size of output_layers: (5, 5, 60, 60)

batch_size = 1
sequence_length_output = 5 * 5

output_layers_with_batch = np.reshape(output_layers, (batch_size, 5, 5, 60, 60))
# Size of output_layers_with_batch: (1, 5, 5, 60, 60)

output_layers_processed = output_layers_with_batch.reshape(batch_size, sequence_length_output, grid_size * grid_size)
# Size of output_layers_processed: (1, 25, 3600)

# Convert to PyTorch tensors
output_tensor_train = torch.tensor(output_layers_processed, dtype=torch.float32)
input_tensor_train = torch.zeros_like(output_tensor_train)
# Size of input_tensor_train: torch.Size([1, 25, 3600])
# Size of output_tensor_train: torch.Size([1, 25, 3600])
print(input_tensor_train.shape)
print(output_tensor_train.shape)

# Initialize model and train
model = MineralTransformer(d_model=d_model)
train(model, input_tensor_train, output_tensor_train, num_epochs=30000, learning_rate=0.001)

model.eval()
with torch.no_grad():
    predicted_output_train, _ = model(input_tensor_train, output_tensor_train)

# Reshape tensors back to the original shape for visualization and metric computation
output_np_train = output_tensor_train.numpy().reshape(batch_size, 5, 5, grid_size, grid_size)
predicted_np_train = predicted_output_train.numpy().reshape(batch_size, 5, 5, grid_size, grid_size)

flattened_output_np_train = output_np_train.reshape(batch_size, 5, 5, -1)
flattened_predicted_np_train = predicted_np_train.reshape(batch_size, 5, 5, -1)
# (1, 5, 5, 3600)

# Logging metric values to wandb
metric_values = []
for i in range(5):
    predicted_sum = np.sum(flattened_predicted_np_train[0, :, i, :])
    ground_truth_sum = np.sum(flattened_output_np_train[0, :, i, :])
    metric = predicted_sum / (ground_truth_sum + 1e-8)
    metric_values.append(metric)
    wandb.log({f'Layer_{chr(65+i)}_Metric': metric})
    print(f'Layer {chr(65+i)}: Metric = {metric}')

plot_metric(metric_values, [chr(65+i) for i in range(5)])
wandb.save('metric_plot.png')  # Save the plot to wandb

# VIS
input_elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper']
output_elements = ['Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']

for i in range(5):  # By quality
    visualize_layers(i, np.zeros_like(output_np_train[0]), output_np_train[0], predicted_np_train[0], input_elements, output_elements)

# Dice coefficients for each layer in the US
dice_coeffs = compute_dice_coefficients(flattened_predicted_np_train[0], flattened_output_np_train[0], threshold=0)
plot_dice_coefficients(dice_coeffs, output_elements)
wandb.save('dice_coefficients_plot.png')  # Save the plot to wandb

# Finish the wandb run
wandb.finish()
'''

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
