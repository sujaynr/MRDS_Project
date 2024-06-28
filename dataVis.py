#Imports:
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

grid_size = 50

# Load data from HDF5 file
h5_file_path = 'prepared_data_TILES/NEmineral_data.h5'

# Load counts data to find non-empty squares
with h5py.File(h5_file_path, 'r') as f:
    counts = f['counts'][:]

print(f"Original shape of dataset: {counts.shape}")

# Layer mapping
elements = ['Gold', 'Silver', 'Zinc', 'Lead', 'Copper', 'Nickel', 'Iron', 'Uranium', 'Tungsten', 'Manganese']
input_minerals = [0, 1, 2, 3, 4, 6, 7, 8, 9]
output_mineral = 5 

print("Layer mapping to elements:")
for i, element in enumerate(elements):
    print(f"Layer {i}: {element}")

pre_training_vis_path = "preTrainingVis"
if not os.path.exists(pre_training_vis_path):
    os.makedirs(pre_training_vis_path)


nickel_counts = np.sum(counts[:, output_mineral, :, :], axis=(1, 2))
other_counts = np.sum(counts[:, input_minerals, :, :], axis=(2, 3))

# Filter to only include squares with non-zero Nickel counts
non_zero_nickel_indices = np.where(nickel_counts > 0)[0]

np.random.seed(0)
random_indices = np.random.choice(non_zero_nickel_indices, size=5, replace=False)

for i, mineral_name in enumerate(elements):
    for idx in random_indices:
        nickel_grid = counts[idx, output_mineral, :, :]
        mineral_grid = counts[idx, i, :, :]
        
        # Get the coordinates and counts of non-zero values
        nickel_coords = np.argwhere(nickel_grid > 0)
        nickel_values = nickel_grid[nickel_grid > 0]
        mineral_coords = np.argwhere(mineral_grid > 0)
        mineral_values = mineral_grid[mineral_grid > 0]
        
        # Convert coordinates to tuples for easy comparison
        nickel_coords_tuples = [tuple(coord) for coord in nickel_coords]
        mineral_coords_tuples = [tuple(coord) for coord in mineral_coords]
        
        # Find overlapping coordinates
        overlap_coords = np.array([coord for coord in nickel_coords_tuples if coord in mineral_coords_tuples])
        
        plt.figure(figsize=(10, 8))
        
        # Plot Nickel points
        scatter = plt.scatter(nickel_coords[:, 1], nickel_coords[:, 0], alpha=0.5, label='Nickel', c=nickel_values, cmap='Reds', edgecolor='k')
        cbar = plt.colorbar(scatter, label='Nickel Count')
        
        # Plot other mineral points
        scatter = plt.scatter(mineral_coords[:, 1], mineral_coords[:, 0], alpha=0.5, label=mineral_name, c=mineral_values, cmap='Blues', edgecolor='k')
        cbar = plt.colorbar(scatter, label=f'{mineral_name} Count')
        
        # Plot overlapping points
        if overlap_coords.size > 0:
            plt.scatter(overlap_coords[:, 1], overlap_coords[:, 0], alpha=0.7, label='Overlap', c='black', marker='x', edgecolor='k', s=100)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.title(f'Scatter plot of Nickel vs {mineral_name} Locations\nSquare Index: {idx}')
        plt.legend()
        plt.grid(True)  # Add grid lines
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
        plt.savefig(f"{pre_training_vis_path}/Nickel_vs_{mineral_name}_Square_{idx}.png")
        plt.close()

print("Pre-training visualizations saved in the 'preTrainingVis' folder.")


total_squares = counts.shape[0]
percent_only_mineral_list = []
percent_mineral_with_nickel_list = []
percent_all_mineral_list = []
total_mineral_points_list = []

for i, mineral_name in enumerate(elements):
    mineral_index = i
    
    # Identify squares that contain the current mineral
    squares_with_mineral = counts[:, mineral_index, :, :] > 0
    valid_squares_with_mineral = np.any(squares_with_mineral, axis=(1, 2))
    
    # Calculate percentage of squares that contain only this mineral (excluding Nickel)
    only_mineral_squares = np.sum(valid_squares_with_mineral & (np.sum(counts[:, output_mineral, :, :] > 0, axis=(1, 2)) == 0))
    percent_only_mineral = (only_mineral_squares / np.sum(valid_squares_with_mineral)) * 100
    percent_only_mineral_list.append(percent_only_mineral)
    
    # Calculate percentage of squares that include both the mineral and Nickel
    mineral_with_nickel_squares = np.sum(valid_squares_with_mineral & np.any(counts[:, output_mineral, :, :] > 0, axis=(1, 2)))
    percent_mineral_with_nickel = (mineral_with_nickel_squares / np.sum(valid_squares_with_mineral)) * 100
    percent_mineral_with_nickel_list.append(percent_mineral_with_nickel)

    # Calculate percentage of all squares that contain the current mineral
    all_mineral_squares = np.sum(valid_squares_with_mineral)
    percent_all_mineral = (all_mineral_squares / total_squares) * 100
    percent_all_mineral_list.append(percent_all_mineral)

    # Calculate the total number of points for the current mineral
    total_mineral_points = np.sum(counts[:, mineral_index, :, :])
    
    # Print results
    print(f"Mineral: {mineral_name}")
    print(f"  Percentage of squares that contain only {mineral_name} (excluding Nickel): {percent_only_mineral:.2f}%")
    print(f"  Percentage of squares that include both {mineral_name} and Nickel: {percent_mineral_with_nickel:.2f}%")
    print(f"  Percentage of all squares that contain {mineral_name}: {percent_all_mineral:.2f}%")
    print(f"  Total number of {mineral_name} points: {total_mineral_points}")

print("Calculation complete.")

# Plotting the results
def plot_results(elements, percent_only_mineral, percent_with_nickel, percent_all_mineral):
    x = np.arange(len(elements))
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width, percent_only_mineral, width, label='Only Mineral %')
    bars2 = ax.bar(x, percent_with_nickel, width, label='Mineral with Nickel %')
    bars3 = ax.bar(x + width, percent_all_mineral, width, label='All Mineral %')

    ax.set_xlabel('Minerals')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of Squares by Mineral')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend(loc='upper left')

    # Add labels above the bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('mineral_analysis.png', bbox_inches='tight')
    plt.show()

# Call the plot function
plot_results(elements, percent_only_mineral_list, percent_mineral_with_nickel_list, percent_all_mineral_list)

print("Plots saved and displayed.")

