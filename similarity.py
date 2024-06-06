import pickle
import numpy as np

import matplotlib.pyplot as plt

elements = ['Gold', 'Silver', 'Nickel', 'Zinc', 'Iron', 'Uranium']
data = np.zeros((len(elements), 5), dtype=object)
dice_sorensen_matrix = np.zeros((len(elements), len(elements), 5))

for i, elem in enumerate(elements):
    for j in range(5):
        with open(f'prepared_data/{elem}_layers(100%).pkl', 'rb') as f:
            elem_data = pickle.load(f)
        data[i, j] = elem_data[j]

for i in range(5):
    print(f"Quality level {chr(65+i)}:")
    for j in range(len(elements)):
        print(f"{elements[j]}: {data[j, i].shape}")
    print()

for k in range(5):
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i == j:
                dice_sorensen_matrix[i, j, k] = 1.0
            else:
                intersection = np.sum(np.minimum(data[i, k], data[j, k]))
                union = np.sum(data[i, k]) + np.sum(data[j, k])
                dice_sorensen_matrix[i, j, k] = 2.0 * intersection / union


fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for k in range(5):
    ax = axs[k]
    im = ax.imshow(dice_sorensen_matrix[:, :, k], cmap='viridis', interpolation='nearest')
    ax.set_title(f"Dice-Sorensen Matrix: ({chr(65+k)})")
    ax.set_xticks(np.arange(len(elements)))
    ax.set_yticks(np.arange(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_yticklabels(elements)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(elements)):
        for j in range(len(elements)):
            text = ax.text(j, i, f"{dice_sorensen_matrix[i, j, k]:.2f}", ha="center", va="center", color="w")

plt.tight_layout()
plt.savefig('mineralSimilarity.png')
plt.show()
