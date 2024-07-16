import wandb
import os
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import structural_similarity_index_measure as ssim

from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def absolute_difference_integral(predicted, target):
    predicted_sum = torch.sum(predicted[:, 0, :, :], dim=[1, 2]).to(target.device)
    target_sum = torch.sum(target[:, 0, :, :], dim=[1, 2])
    diff_integral = torch.abs(predicted_sum - target_sum).mean()
    return diff_integral


def masked_mse_loss(predicted, target, mask, include_true_negatives=False):
    mask = mask.to(predicted.device)  # Ensure mask is on the same device as predicted and target
    target = target.to(predicted.device)  # Ensure target is on the same device as predicted
    
    if include_true_negatives:
        # Apply the mask to the predicted and target tensors
        masked_predicted = predicted * mask
        masked_target = target * mask
    else:
        # Use only non-empty cells for loss calculation
        mask = target[:, 0, :, :] > 0
        masked_predicted = predicted[:, 0, :, :][mask]
        masked_target = target[:, 0, :, :][mask]
    
    # Calculate MSE loss only on the masked regions
    loss = nn.MSELoss()(masked_predicted, masked_target)
    return loss

def combined_loss(predicted, target, first_loss_type, include_true_negatives=False, lagrange_multiplier_pixel=1e7, lagrange_multiplier_integral=1e-5):
    """
    Combined loss function.
    If the first loss type is pixel, then the combined loss is:
        integral_loss(predicted, target) + lagrange_multiplier_pixel * masked_mse_loss(predicted, target, create_nonzero_mask(target), include_true_negatives=include_true_negatives)
    If the first loss type is integral, then the combined loss is:
        masked_mse_loss(predicted, target, create_nonzero_mask(target), include_true_negatives=include_true_negatives) + lagrange_multiplier_integral * integral_loss(predicted, target)
    """
    target = target.to(predicted.device)  # Ensure target is on the same device as predicted
    
    if first_loss_type == 'pixel':
        primary_loss_value = masked_mse_loss(predicted, target, create_nonzero_mask(target).to(predicted.device), include_true_negatives=include_true_negatives)
        secondary_loss_value = integral_loss(predicted, target)
        return secondary_loss_value + lagrange_multiplier_pixel * primary_loss_value
    elif first_loss_type == 'integral':
        primary_loss_value = integral_loss(predicted, target)
        secondary_loss_value = masked_mse_loss(predicted, target, create_nonzero_mask(target).to(predicted.device), include_true_negatives=include_true_negatives)
        return secondary_loss_value + lagrange_multiplier_integral * primary_loss_value
    else:
        raise ValueError(f"Unknown first_loss_type: {first_loss_type}")


def create_nonzero_mask(mineral_data):
    # Create a mask where cells with any nonzero value in any mineral layer are marked as 1, others as 0
    nonzero_mask = (mineral_data.sum(axis=1) > 0).float()
    return nonzero_mask


def dice_coefficient_nonzero(pred, target, threshold=0.0, smooth=1e-6):
    pred = (pred > threshold).float().to(target.device)
    target = (target > threshold).float().to(target.device)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    nonzero_mask = (pred_flat + target_flat) > 0

    pred_nonzero = pred_flat[nonzero_mask]
    target_nonzero = target_flat[nonzero_mask]

    intersection = (pred_nonzero * target_nonzero).sum()

    dice = (2. * intersection + smooth) / (pred_nonzero.sum() + target_nonzero.sum() + smooth)
    return dice

def plot_predictions(predicted, ground_truth, input_data, input_minerals, fault_data, geo_age_data, elevation_data, output_mineral_name, num_samples=20, save_path="FULLSWEEP1", specs="NONE"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese', 'Fault', 'Min Age', 'Max Age', 'Rocktype', 'Elevation']

    non_empty_output_indices = np.where(np.sum(ground_truth[:, 0, :, :], axis=(1, 2)) > 0)[0]
    indices = random.sample(list(non_empty_output_indices), min(num_samples, len(non_empty_output_indices)))

    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(3, 5, figsize=(30, 15))
        predicted_image = predicted[idx, 0, :, :]
        ground_truth_image = ground_truth[idx, 0, :, :]
        
        im1 = axes[0, 0].imshow(predicted_image, cmap='viridis')
        pred_sum = predicted_image.sum()
        axes[0, 0].set_title(f'Predicted {output_mineral_name}\nTotal: {pred_sum:.2f}')
        
        im2 = axes[0, 1].imshow(ground_truth_image, cmap='viridis')
        gt_sum = ground_truth_image.sum()
        axes[0, 1].set_title(f'Ground Truth {output_mineral_name}\nTotal: {gt_sum:.2f}')
        
        for j, mineral_index in enumerate(input_minerals):
            layer_image = input_data[idx, mineral_index, :, :]
            im = axes[(j + 2) // 5, (j + 2) % 5].imshow(layer_image, cmap='viridis')
            layer_sum = layer_image.sum()
            axes[(j + 2) // 5, (j + 2) % 5].set_title(f'{elements[mineral_index]}\nTotal: {layer_sum:.2f}')
            fig.colorbar(im, ax=axes[(j + 2) // 5, (j + 2) % 5])

        fault_image = fault_data[idx, 0, :, :]
        im3 = axes[2, 2].imshow(fault_image, cmap='viridis')
        fault_sum = fault_image.sum()
        axes[2, 2].set_title(f'Fault Data\nTotal: {fault_sum:.2f}')
        fig.colorbar(im3, ax=axes[2, 2])

        min_age_image = geo_age_data[idx, 0, :, :]
        im4 = axes[2, 3].imshow(min_age_image, cmap='viridis')
        min_age_sum = min_age_image.sum()
        axes[2, 3].set_title(f'Min Age\nTotal: {min_age_sum:.2f}')
        fig.colorbar(im4, ax=axes[2, 3])

        max_age_image = geo_age_data[idx, 1, :, :]
        im5 = axes[2, 4].imshow(max_age_image, cmap='viridis')
        max_age_sum = max_age_image.sum()
        axes[2, 4].set_title(f'Max Age\nTotal: {max_age_sum:.2f}')
        fig.colorbar(im5, ax=axes[2, 4])

        rocktype_image = geo_age_data[idx, 2, :, :]
        im6 = axes[2, 1].imshow(rocktype_image, cmap='viridis')
        rocktype_sum = rocktype_image.sum()
        axes[2, 1].set_title(f'Rocktype\nTotal: {rocktype_sum:.2f}')
        fig.colorbar(im6, ax=axes[2, 1])

        elevation_image = elevation_data[idx, :, :]
        im7 = axes[2, 0].imshow(elevation_image, cmap='viridis')
        elevation_sum = elevation_image.sum()
        axes[2, 0].set_title(f'Elevation\nTotal: {elevation_sum:.2f}')
        fig.colorbar(im7, ax=axes[2, 0])

        dice_value = dice_coefficient_nonzero(torch.tensor(predicted_image).unsqueeze(0).unsqueeze(0), 
                                              torch.tensor(ground_truth_image).unsqueeze(0).unsqueeze(0)).item()
        fig.suptitle(f'Sample {i} - Dice Coefficient: {dice_value:.4f}', fontsize=16)
        
        fig.colorbar(im1, ax=axes[0, 0])
        fig.colorbar(im2, ax=axes[0, 1])
        
        plt.savefig(f"{save_path}/{output_mineral_name}_{specs}_sample_{i}.png")
        plt.close()

def binary_cross_entropy_loss(predicted, target):
    return nn.BCELoss()(predicted, target)

def custom_loss(predicted, target, alpha=1.0, beta=0.01):
    mse_loss = F.mse_loss(predicted, target)
    # Zero Variance Penalty: penalize low variance in predictions
    variance_penalty = torch.var(predicted, dim=[1, 2, 3]).mean()
    loss = alpha * mse_loss + beta / variance_penalty
    return loss

def regular_loss(predicted, target):
    return nn.MSELoss()(predicted[:, 0, :, :], target[:, 0, :, :])

def integral_loss(predicted, target):  # MSE ON SUMS
    predicted_sum = torch.sum(predicted[:, 0, :, :], dim=[1, 2]).to(target.device)
    target_sum = torch.sum(target[:, 0, :, :], dim=[1, 2])
    return nn.MSELoss()(predicted_sum, target_sum)


def nonempty_loss(predicted, target):  # MSE PER PIXEL FOR NON-EMPTY
    mask = target[:, 0, :, :] > 0 
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predicted.device)
    return nn.MSELoss()(predicted[:, 0, :, :][mask], target[:, 0, :, :][mask])

def weighted_mse_loss(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)

def plot_metric(metric_values, layers):
    x = np.arange(len(layers))
    plt.bar(x, metric_values)
    plt.xticks(x, layers)
    plt.title('Custom Metric')
    plt.xlabel('Layer')
    plt.ylabel('Metric Value')
    plt.savefig('trainingVis/metric_plot.png')
    plt.close()


def gaussian_smooth_and_normalize(layers, sigma=1.0):
    smoothed_layers = []
    for layer in layers:
        smoothed_layer = gaussian_filter(layer, sigma=sigma)
        normalized_layer = smoothed_layer * (layer.sum() / smoothed_layer.sum())
        smoothed_layers.append(normalized_layer)
    return np.stack(smoothed_layers, axis=0)

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

def calculate_dice_coefficient(pred, gt, threshold=0.0):
    pred_binary = pred > threshold
    gt_binary = gt > threshold
    
    intersection = np.sum(pred_binary & gt_binary)
    pred_sum = np.sum(pred_binary)
    gt_sum = np.sum(gt_binary)
    
    dice = (2. * intersection) / (pred_sum + gt_sum)
    return dice


def compute_dice_coefficients(smoothed_predicted_np_train, smoothed_output_np_train, threshold=0.0): # TUNE
    dice_coeffs = {}
    
    for layer_index in range(5):
        dice_per_layer = []
        for i in range(smoothed_predicted_np_train.shape[0]):  # For each element
            dice_coeff = calculate_dice_coefficient(smoothed_predicted_np_train[i][layer_index], smoothed_output_np_train[i][layer_index], threshold)
            dice_per_layer.append(dice_coeff)
            print(f'Layer {chr(65+layer_index)}, Element {smoothed_output_np_train[i]}: Dice Coefficient = {dice_coeff}')
        
        dice_coeffs[layer_index] = dice_per_layer
    
    return dice_coeffs

def plot_dice_coefficients(dice_coeffs, output_elements):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    fig.suptitle('Dice Coefficients for Each Quality Layer', fontsize=16)
    
    for layer_index, dice_per_layer in dice_coeffs.items():
        ax = axes[layer_index]
        ax.bar(output_elements, dice_per_layer)
        ax.axhline(y=np.mean(dice_per_layer), color='r', linestyle='--') # Mean
        ax.set_title(f'Quality Layer {chr(65+layer_index)}')
        ax.set_xlabel('Mineral')
        ax.set_ylabel('Dice Coefficient')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join('trainingVis', 'dice_coefficients.png'))
    plt.show()
