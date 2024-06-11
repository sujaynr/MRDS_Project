import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader, Dataset


# Set the conda environment
conda_env = '/Users/sujaynair/anaconda3/envs/dataAnalysis'
os.environ['CONDA_PREFIX'] = conda_env
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ElementQualityTransformer(nn.Module):
    def __init__(self, spatial_dim=900, quality_dim=5, element_dim=100, d_model=None, nhead=8, num_layers=6):
        super(ElementQualityTransformer, self).__init__()
        if d_model is None:
            d_model = (spatial_dim * quality_dim + element_dim + nhead - 1) // nhead * nhead  # Ensure divisibility by nhead

        self.spatial_embedding = nn.Linear(spatial_dim * quality_dim, d_model)
        self.element_embedding = nn.Embedding(100, element_dim)  # Assuming 100 unique elements
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, spatial_dim * quality_dim)  # Project back to the original size

    def forward(self, input_elements, input_grids, output_elements):
        print(input_elements.shape, input_grids.shape, output_elements.shape)
        
        # Ensure input_grids has shape (batch_size, n, 5, 30, 30)
        batch_size, n, q, h, w = input_grids.shape
        assert q == 5 and h == 30 and w == 30, f"input_grids has incorrect shape: {input_grids.shape}"
        
        m = output_elements.shape[0]  # Number of output elements

        spatial_dim = 900  # Flattened 30x30
        element_dim = 100  # Embedding dimension for elements
        d_model = self.encoder_layer.self_attn.embed_dim  # Get the d_model from encoder layer

        # Flatten and embed input grids
        input_grids = input_grids.view(batch_size, n, q * h * w)  # (batch_size, n, 5 * 30 * 30)
        print(input_grids.shape)
        spatial_embeds = self.spatial_embedding(input_grids)  # (batch_size, n, d_model)
        print(spatial_embeds.shape)

        assert spatial_embeds.shape == (batch_size, n, d_model), f"spatial_embeds has incorrect shape: {spatial_embeds.shape}"

        # Embed elements
        input_element_embeds = self.element_embedding(input_elements)  # (batch_size, n, element_dim)

        assert input_element_embeds.shape == (batch_size, n, element_dim), f"input_element_embeds has incorrect shape: {input_element_embeds.shape}"

        # Combine embeddings
        input_embeds = torch.cat((spatial_embeds, input_element_embeds), dim=-1)  # (batch_size, n, d_model + element_dim)

        assert input_embeds.shape == (batch_size, n, d_model + element_dim), f"input_embeds has incorrect shape: {input_embeds.shape}"

        # Encoder
        encoder_output = self.encoder(input_embeds)  # (batch_size, n, d_model + element_dim)

        assert encoder_output.shape == (batch_size, n, d_model + element_dim), f"encoder_output has incorrect shape: {encoder_output.shape}"

        # Prepare target embeddings for decoder
        output_element_embeds = self.element_embedding(output_elements).unsqueeze(1)  # (batch_size, 1, element_dim)
        output_element_embeds = output_element_embeds.repeat(1, m, 1)  # (batch_size, m, element_dim)

        assert output_element_embeds.shape == (batch_size, m, element_dim), f"output_element_embeds has incorrect shape: {output_element_embeds.shape}"

        # Create decoder input embeddings (can be zero-initialized or learned embeddings for start tokens)
        decoder_input_embeds = torch.zeros((batch_size, m, d_model), device=input_elements.device)  # (batch_size, m, d_model)
        decoder_input_embeds[:, :, :element_dim] = output_element_embeds  # Use element embeddings for decoder inputs

        assert decoder_input_embeds.shape == (batch_size, m, d_model), f"decoder_input_embeds has incorrect shape: {decoder_input_embeds.shape}"

        # Decoder
        decoder_output = self.decoder(decoder_input_embeds, encoder_output)  # (batch_size, m, d_model)

        assert decoder_output.shape == (batch_size, m, d_model), f"decoder_output has incorrect shape: {decoder_output.shape}"

        # Project to the original concatenated size
        output = self.output_projection(decoder_output)  # (batch_size, m, spatial_dim * quality_dim)

        assert output.shape == (batch_size, m, spatial_dim * quality_dim), f"output has incorrect shape: {output.shape}"

        # Reshape to (batch_size, m, 5, 30, 30)
        output = output.view(batch_size, m, 5, 30, 30)

        assert output.shape == (batch_size, m, 5, 30, 30), f"output has incorrect shape: {output.shape}"

        return output

# Data loading
data_dir = 'prepared_data'
elements = ['Gold', 'Silver', 'Nickel']
data = {}

for elem in elements:
    with open(os.path.join(data_dir, f'{elem}_layers(100%).pkl'), 'rb') as f:
        data[elem] = pickle.load(f)

def prepare_data(data):
    prepared_data = []
    gold_data = data['Gold']
    silver_data = data['Silver']
    nickel_data = data['Nickel']

    assert gold_data.shape == silver_data.shape == nickel_data.shape, "All data elements must have the same shape"
    assert gold_data.shape == (5, 30, 30), "Data elements must have the shape (5, 30, 30)"

    for i in range(gold_data.shape[1]):  # Iterate over the second dimension (30)
        for j in range(gold_data.shape[2]):  # Iterate over the third dimension (30)
            input_elements = torch.tensor([0, 1], dtype=torch.long)  # Indices for Gold and Silver
            
            gold_grid = torch.tensor(gold_data[:, i, j], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape (1, 5, 1, 1)
            silver_grid = torch.tensor(silver_data[:, i, j], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape (1, 5, 1, 1)
            nickel_grid = torch.tensor(nickel_data[:, i, j], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # Shape (5, 1, 1)

            input_grids = torch.cat([gold_grid, silver_grid], dim=0)  # Shape (2, 5, 1, 1)
            input_grids = input_grids.repeat(1, 1, 30, 30)  # Shape (2, 5, 30, 30)

            assert input_grids.shape == (2, 5, 30, 30), f"input_grids shape mismatch: {input_grids.shape}"

            input_grids = input_grids.unsqueeze(0)  # Add batch dimension, shape (1, 2, 5, 30, 30)
            
            assert input_grids.shape == (1, 2, 5, 30, 30), f"input_grids shape mismatch after unsqueeze: {input_grids.shape}"

            output_elements = torch.tensor([2], dtype=torch.long)  # Index for Nickel
            output_grids = nickel_grid.repeat(1, 30, 30)  # Shape (5, 30, 30)
            output_grids = output_grids.unsqueeze(0)  # Shape (1, 5, 30, 30)

            assert output_grids.shape == (1, 5, 30, 30), f"output_grids shape mismatch: {output_grids.shape}"

            prepared_data.append((input_elements, input_grids, output_elements, output_grids))
    
    return prepared_data

prepared_data = prepare_data(data)
pdb.set_trace()

class ElementQualityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_elements, input_grids, output_elements, output_grids = self.data[idx]
        return (input_elements, input_grids, output_elements, output_grids)



def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (input_elements, input_grids, output_elements, output_grids) in enumerate(dataloader):
            # Move tensors to the appropriate device
            input_elements = input_elements.to(device)
            input_grids = input_grids.to(device)
            output_elements = output_elements.to(device)
            output_grids = output_grids.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_elements, input_grids, output_elements)

            # Compute the loss
            loss = criterion(outputs, output_grids)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')

# Create the dataset and dataloader
dataset = ElementQualityDataset(prepared_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ElementQualityTransformer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=10)
