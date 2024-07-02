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

def dice_coefficient(preds, target, threshold=0.5, smooth=1e-6):
    # Binarize at first
    preds = (preds > threshold).float()
    target = (target > threshold).float()

    preds_flat = preds.view(-1)
    target_flat = target.view(-1)

    intersection = (preds_flat * target_flat).sum()
    union = preds_flat.sum() + target_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def plot_predictions(predicted, ground_truth, input_data, input_minerals, output_mineral_name, num_samples=20, save_path="SweepVISB1", specs="NONE"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
    
    # Find indices where the output mineral layer is non-empty
    non_empty_output_indices = np.where(np.sum(ground_truth[:, 0, :, :], axis=(1, 2)) > 0)[0]
    indices = random.sample(list(non_empty_output_indices), min(num_samples, len(non_empty_output_indices)))
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(2, 6, figsize=(30, 10))
        
        predicted_image = predicted[idx, 0, :, :]
        ground_truth_image = ground_truth[idx, 0, :, :]
        
        # Plot predicted output mineral
        im1 = axes[0, 0].imshow(predicted_image, cmap='viridis')
        pred_sum = predicted_image.sum()
        axes[0, 0].set_title(f'Predicted {output_mineral_name}\nTotal: {pred_sum:.2f}')
        
        # Plot ground truth output mineral
        im2 = axes[0, 1].imshow(ground_truth_image, cmap='viridis')
        gt_sum = ground_truth_image.sum()
        axes[0, 1].set_title(f'Ground Truth {output_mineral_name}\nTotal: {gt_sum:.2f}')
        
        # Plot ground truth for other 9 layers
        for j, mineral_index in enumerate(input_minerals):
            layer_image = input_data[idx, mineral_index, :, :]
            im = axes[(j + 2) // 6, (j + 2) % 6].imshow(layer_image, cmap='viridis')
            layer_sum = layer_image.sum()
            axes[(j + 2) // 6, (j + 2) % 6].set_title(f'{elements[mineral_index]}\nTotal: {layer_sum:.2f}')
            fig.colorbar(im, ax=axes[(j + 2) // 6, (j + 2) % 6])


        dice_value = dice_coefficient(torch.tensor(predicted_image).unsqueeze(0).unsqueeze(0), 
                                      torch.tensor(ground_truth_image).unsqueeze(0).unsqueeze(0)).item()
        fig.suptitle(f'Sample {i} - Dice Coefficient: {dice_value:.4f}', fontsize=16)
        
        fig.colorbar(im1, ax=axes[0, 0])
        fig.colorbar(im2, ax=axes[0, 1])
        

        plt.savefig(f"{save_path}/{output_mineral_name}_{specs}_sample_{i}.png")
        plt.close()

def custom_loss(predicted, target, alpha=1.0, beta=0.01):
    mse_loss = F.mse_loss(predicted, target)
    # Zero Variance Penalty: penalize low variance in predictions
    variance_penalty = torch.var(predicted, dim=[1, 2, 3]).mean()
    loss = alpha * mse_loss + beta / variance_penalty
    return loss

def regular_loss(predicted, target):  # MSE PER PIXEL
    return nn.MSELoss()(predicted[:, 0, :, :], target[:, 0, :, :])

def integral_loss(predicted, target):  # MSE ON SUMS
    predicted_sum = torch.sum(predicted[:, 0, :, :], dim=[1, 2])
    target_sum = torch.sum(target[:, 0, :, :], dim=[1, 2])
    return nn.MSELoss()(predicted_sum, target_sum)

def combined_loss(predicted, target, first_loss_type, lagrange_multiplier_pixel=1e7, lagrange_multiplier_integral=1e-5):
    """
    Combined loss function.
    If the first loss type is pixel, then the combined loss is:
        integral_loss(predicted, target) + lagrange_multiplier_pixel * pixel_loss(predicted, target)
    If the first loss type is integral, then the combined loss is:
        pixel_loss(predicted, target) + lagrange_multiplier_integral * integral_loss(predicted, target)
    """
    if first_loss_type == 'pixel':
        primary_loss_value = regular_loss(predicted, target)
        secondary_loss_value = integral_loss(predicted, target)
        return secondary_loss_value + lagrange_multiplier_pixel * primary_loss_value
    elif first_loss_type == 'integral':
        primary_loss_value = integral_loss(predicted, target)
        secondary_loss_value = regular_loss(predicted, target)
        return secondary_loss_value + lagrange_multiplier_integral * primary_loss_value
    else:
        raise ValueError(f"Unknown first_loss_type: {first_loss_type}")


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    predicted_output = []
    output_tensor = []

    with torch.no_grad():
        for input_tensor, output in data_loader:
            input_tensor = input_tensor.to(device)
            output = output.to(device)
            outputs = model(input_tensor)

            loss = criterion(outputs, output)
            # pdb.set_trace()
            total_loss += loss.item()

            predicted_output.append(outputs)
            output_tensor.append(output)

    avg_loss = total_loss / len(data_loader)
    predicted_output = torch.cat(predicted_output, dim=0)
    output_tensor = torch.cat(output_tensor, dim=0)

    return avg_loss, predicted_output, output_tensor

def train(model, train_loader, test_loader, num_epochs=50, learning_rate=0.0001, two_step=False, first_loss='integral'):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer
    
    if first_loss == 'integral':
        primary_loss = integral_loss
    else:
        primary_loss = regular_loss

    losses = []  # List to store training losses
    test_losses = []  # List to store test losses

    # First step of training
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0  # Initialize total loss
        
        for input_tensor_train, output_tensor_train in train_loader:
            optimizer.zero_grad()  # Zero gradients
            outputs = model(input_tensor_train.to(device))  # Pass only the input tensor
            loss = primary_loss(outputs, output_tensor_train.to(device))  # Calculate loss
            loss.backward()  # Backpropagate loss

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss

        avg_loss = total_loss / len(train_loader)  # Average loss
        losses.append(avg_loss)  # Store average loss

        test_loss, _, _ = evaluate(model, test_loader, primary_loss)  # Evaluate model on test set
        test_losses.append(test_loss)  # Store test loss

        wandb.log({"Train Loss (Step 1)": avg_loss, "Test Loss (Step 1)": test_loss})  # Log losses to wandb

        print(f'Epoch {epoch+1}/{num_epochs} (Step 1), Train Loss: {avg_loss}, Test Loss: {test_loss}')  # Print losses

    # Second step of training (if two-step training is enabled)
    if two_step:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            total_loss = 0  # Initialize total loss
            
            for input_tensor_train, output_tensor_train in train_loader:
                optimizer.zero_grad()  # Zero gradients
                outputs = model(input_tensor_train.to(device))  # Pass only the input tensor
                loss = combined_loss(outputs, output_tensor_train.to(device), first_loss)  # Calculate combined loss
                loss.backward()  # Backpropagate loss

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()  # Update weights

                total_loss += loss.item()  # Accumulate loss

            avg_loss = total_loss / len(train_loader)  # Average loss
            losses.append(avg_loss)  # Store average loss

            test_loss, _, _ = evaluate(model, test_loader, lambda p, t: combined_loss(p, t, first_loss))  # Evaluate model on test set
            test_losses.append(test_loss)  # Store test loss

            wandb.log({"Train Loss (Step 2)": avg_loss, "Test Loss (Step 2)": test_loss})  # Log losses to wandb

            print(f'Epoch {epoch+1}/{num_epochs} (Step 2), Train Loss: {avg_loss}, Test Loss: {test_loss}')  # Print losses

    # Evaluate final model on test set
    final_criterion = lambda p, t: combined_loss(p, t, first_loss) if two_step else primary_loss(p, t)
    test_loss, predicted_output_test, output_tensor_test = evaluate(model, test_loader, final_criterion)

    print(f'Final Test Loss: {test_loss}')

    # Plot training and test loss curves
    plt.plot(losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('trainingVis/loss_curve.png')
    plt.close()
    
    wandb.save('trainingVis/loss_curve.png')  # Save loss curve plot to wandb

    return predicted_output_test, output_tensor_test 

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
