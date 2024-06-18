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
