import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


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


def calculate_dice_coefficient(pred, gt, threshold=0.05):
    pred_binary = pred > threshold
    gt_binary = gt > threshold
    
    intersection = np.sum(pred_binary & gt_binary)
    pred_sum = np.sum(pred_binary)
    gt_sum = np.sum(gt_binary)
    
    dice = (2. * intersection) / (pred_sum + gt_sum)
    return dice


def compute_dice_coefficients(smoothed_predicted_np_train, smoothed_output_np_train, threshold=0.05):
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
