import os
import subprocess

# Define hyperparameters
learning_rates = [0.0001, 0.001]  # List of learning rates to sweep
batch_size = 64                   # Fixed batch size
model_type = 'u'                  # Fixed model type
minerals = ['Nickel', 'Gold']     # List of minerals
seeds = [123, 124, 125, 126, 127, 128]           # List of seeds
epochs = 30                       # Number of epochs
use_raca_flags = [True, False]    # Whether to use RaCA data or not

# Fixed loss scheme
loss_scheme = {'two_step': True, 'first_loss': 'integral'}

# Generate commands
commands = []
for use_raca in use_raca_flags:
    for seed in seeds:
        for mineral in minerals:
            for lr in learning_rates:
                cmd = [
                    "python", "train.py",
                    "--learning_rate", str(lr),
                    "--batch_size", str(batch_size),
                    "--model_type", model_type,
                    "--output_mineral_name", mineral,
                    "--loss1", loss_scheme['first_loss'],
                    "--two_step", str(loss_scheme['two_step']),
                    "--logName", f"4FULLRACASWEEP_OCT_lr{lr}_bs{batch_size}_mt{model_type}_loss1:{loss_scheme['first_loss']}_two_step{loss_scheme['two_step']}_use_raca{use_raca}_seed{seed}_mineral{mineral}_epochs{epochs}",
                    "--set_seed", str(seed),
                    "--num_epochs", str(epochs)
                ]
                if use_raca:
                    cmd.append("--use_raca")
                commands.append(cmd)

# Run the commands
for cmd in commands:
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)
