
#### UNET SWEEP

# import subprocess

# # These are the three flag configurations you want to sweep over
# flag_combos = [
#     ["--use_minerals"],
#     ["--use_minerals", "--use_geophys"],
#     ["--use_minerals", "--use_geophys", "--use_raca_data"]
# ]

# # The four seeds you want to run for each configuration
# seeds = [42, 43, 44, 45]

# # Fixed hyperparameters (customize as you like)
# learning_rate = 1e-3
# raca_dim      = 64
# hidden_dim    = 256
# nhead         = 8
# num_layers    = 2
# arch          = "resnet152"

# run_count = 1

# for combo_idx, flags in enumerate(flag_combos):
#     # Create a short label for logging based on which flags are used
#     if   flags == ["--use_minerals"]:
#         combo_name = "MineralsOnly"
#     elif flags == ["--use_minerals", "--use_geophys"]:
#         combo_name = "MineralsGeo"
#     else:
#         combo_name = "MineralsGeoRaca"
    
#     for seed in seeds:
#         print(f"Running combo: {combo_name}, seed: {seed}")
        
#         # Build the command
#         command = [
#             "python", "train.py",
#             "--grid_size",        "50",
#             "--hidden_dim",       str(hidden_dim),
#             "--num_minerals",     "15",
#             "--nhead",            str(nhead),
#             "--num_layers",       str(num_layers),
#             "--d_model",          "256",
#             "--dropout_rate",     "0.2",
#             "--model_type",       "u",
#             "--unetArch",         arch,              # fixed to resnet152
#             "--output_mineral_name", "all",
#             "--learning_rate",    str(learning_rate),
#             "--raca_odim",        str(raca_dim),
#             "--num_epochs",       "300",
#             "--loss1",            "integral",
#             "--batch_size",       "128",
#             "--set_seed",         str(seed),
#             "--logName",
#             f"3SAVEFOREVAL_Run{run_count}_{combo_name}_Seed{seed}"
#         ] + flags  # Add the flags for this combo
        
#         print("Command:", " ".join(command))
#         subprocess.run(command, check=True)
        
#         run_count += 1


#### TRANSFORMER SWEEP
import subprocess

# Define parameter lists in ascending order
hidden_dims = [64] # DO FOR 128, 256
nheads      = [4, 8]
num_layers  = [4]
batch_sizes = [16, 32]
seeds       = [42]

# Fixed hyperparameters (adjust as needed)
learning_rate = 1e-3
raca_dim      = 64
num_epochs    = 300
grid_size     = 50

# Additional flags you always want
base_flags = [
    "--use_minerals",
    "--use_geophys",
    "--use_raca_data"
]

run_count = 1

# Largest to smallest loops, using reversed()
for hd in reversed(hidden_dims):          # 256 -> 128 -> 64
    for nh in reversed(nheads):           # 8 -> 4
        for nl in reversed(num_layers):   # 4 -> 2
            for bs in reversed(batch_sizes):  # 32 -> 16
                for seed in seeds:        # 42, 43, 44, 45
                    log_name = (
                        f"TransformerRun_{run_count}_"
                        f"HD{hd}_NHEAD{nh}_NLAYERS{nl}_BS{bs}_Seed{seed}"
                    )
                    
                    command = [
                        "python", "train.py",
                        "--grid_size",         str(grid_size),
                        "--hidden_dim",        str(hd),
                        "--nhead",             str(nh),
                        "--num_layers",        str(nl),
                        "--d_model",           str(hd),  # if your code needs d_model separately
                        "--learning_rate",     str(learning_rate),
                        "--raca_odim",         str(raca_dim),
                        "--num_epochs",        str(num_epochs),
                        "--batch_size",        str(bs),
                        "--model_type",        "spatial_transformer",  # Key line
                        "--set_seed",          str(seed),
                        "--loss1",             "integral",
                        "--output_mineral_name", "all",
                        "--logName",           log_name
                    ] + base_flags

                    print(f"\n[Run #{run_count}] Command:")
                    print(" ".join(command))
                    
                    # Uncomment to actually run the command
                    subprocess.run(command, check=True)
                    
                    run_count += 1
