#### B1

import subprocess

model_types = ['tc', 'u', 'lc', 'lt']  # tc: TransformerToConv, u: UNet, lc: LinToConv, lt: LinToTransformer
learning_rates = [0.00001, 0.0001]
loss_combinations = ['integral', 'pixel']
output_minerals = ['Gold', 'Nickel']

for model_type in model_types:
    for lr in learning_rates:
        for first_loss in loss_combinations:
            for output_mineral in output_minerals:
                two_step_options = [True, False]
                for two_step in two_step_options:
                    command = [
                        'python', 'train.py',
                        '--model_type', model_type,
                        '--output_mineral_name', output_mineral,
                        '--learning_rate', str(lr),
                        '--num_epochs', '20',
                        '--loss1', first_loss,
                        '--two_step', str(two_step)
                    ]

                    print(f"Running: {' '.join(command)}")
                    subprocess.run(command)