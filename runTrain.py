import subprocess
from utils import create_gif

# Fixed flags for each run
base_flags = ["--use_minerals", "--use_geophys"]

# Learning rates to test
learning_rates = [1e-3, 5.5e-4, 1e-3]

# Run train.py for each learning rate
for i, lr in enumerate(learning_rates, start=1):
    print(f"Running configuration {i}/3 with LR={lr}: {base_flags}")
    command = [
        "python", "train.py",
        "--grid_size", "50",
        "--hidden_dim", "256",
        "--intermediate_dim", "512",
        "--num_minerals", "15",
        "--nhead", "4",
        "--num_layers", "1",
        "--d_model", "256",
        "--dropout_rate", "0.2",
        "--model_type", "u",
        "--output_mineral_name", "all",
        "--learning_rate", str(lr),  # Set learning rate
        "--num_epochs", "750",  # Set epochs
        "--loss1", "integral",
        "--batch_size", "128",
        "--set_seed", "42",
        "--logName", f"BCE_FULL_WITHMASKING(all)_LR{lr}",  # Unique WandB log name
    ] + base_flags

    # Execute the command
    subprocess.run(command, check=True)

# Uncomment if needed for creating GIFs
# image_folder = "/home/sujaynair/MRDS_Project/plotOutputsDec"  # Replace with the path to your folder
# ims = 50
# test_gif_name = f"{ims}test.gif"
# create_gif(image_folder, test_gif_name, "test_", max_images=ims, duration=200)
