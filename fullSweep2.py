import os
import subprocess


learning_rates = [0.00001, 0.0001]
batch_sizes = [16, 64]
model_types = ['u', 'lc']
minerals = ['Gold', 'Nickel']
loss_schemes = [
    {'two_step': True, 'first_loss': 'integral'},
    {'two_step': False, 'first_loss': 'integral'}, 
    {'two_step': False, 'first_loss': 'pixel'}
]
tn_flags = [False, True]
seeds = [456]
epochs = 18

# Generate commands
commands = []
for lr in learning_rates:
    for batch_size in batch_sizes:
        for model_type in model_types:
            for loss_scheme in loss_schemes:
                for tn in tn_flags:
                    for seed in seeds:
                        for mineral in minerals:
                            cmd = [
                                "python", "train.py",
                                "--learning_rate", str(lr),
                                "--batch_size", str(batch_size),
                                "--model_type", model_type,
                                "--output_mineral_name", mineral,
                                "--loss1", loss_scheme['first_loss'],
                                "--two_step", str(loss_scheme['two_step']),
                                "--logName", f"SWEEP1_lr{lr}_bs{batch_size}_mt{model_type}_loss1:{loss_scheme['first_loss']}_two_step{loss_scheme['two_step']}_tn{tn}_seed{seed}_mineral{mineral}_epochs{epochs}", 
                                "--set_seed", str(seed),
                                "--num_epochs", str(epochs)
                            ]
                            if tn:
                                cmd.append("--tn")
                            commands.append(cmd)

for cmd in commands:
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)
