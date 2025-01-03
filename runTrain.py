import subprocess
from utils import create_gif

# # Data source flags
# data_sources = ["--use_geophys", "--use_raca_data"]

# # Generate all combinations of the other flags
# def generate_combinations(sources):
#     combinations = []
#     for i in range(2**len(sources)):  # Include empty set
#         combo = [sources[j] for j in range(len(sources)) if i & (1 << j)]
#         combinations.append(combo)
#     return combinations

# # Ensure --use_minerals is always included
# base_flag = "--use_minerals"
# combinations = generate_combinations(data_sources)

# # Run train.py with each combination of flags
# for i, combo in enumerate(combinations, start=1):
#     full_combo = [base_flag] + combo  # Always include --use_minerals
#     print(f"Running combination {i}/{len(combinations)}: {full_combo}")
#     command = [
#         "python", "train.py",
#         "--grid_size", "50",
#         "--hidden_dim", "256",
#         "--intermediate_dim", "512",
#         "--num_minerals", "15",
#         "--nhead", "4",
#         "--num_layers", "1",
#         "--d_model", "256",
#         "--dropout_rate", "0.2",
#         "--model_type", "u",
#         "--output_mineral_name", "Gold",
#         "--learning_rate", "0.001",
#         "--num_epochs", "100",  # Set epochs 
#         "--loss1", "integral",
#         "--batch_size", "16",
#         "--set_seed", "42",
#         "--logName", f"BCE_FULLPREDICTION(GOLD)_{i}",  # Unique WandB log name
#     ] + full_combo

#     # Execute the command
#     subprocess.run(command, check=True)

image_folder = "/home/sujaynair/MRDS_Project/plotOutputsDec"  # Replace with the path to your folder
# train_gif_name = "train.gif"
ims = 50
test_gif_name = f"{ims}test.gif"

# Create GIFs
# create_gif(image_folder, train_gif_name, "train_", max_images=30, duration=200)
create_gif(image_folder, test_gif_name, "test_", max_images=ims, duration=200)
