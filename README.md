# Mineral Transformer Project

This project aims to develop a transformer-based model to predict the distribution of various minerals across the United States using geospatial data. The model ensures the total number of elements in the E layer remains constant during training and evaluation.

## Model Overview

The `SimplestMineralTransformer` model is built on a transformer architecture designed to handle geospatial data with multiple quality layers (A to E). The model includes:

- Input projection layer
- Transformer encoder
- Transformer decoder
- Output projection layer
- Layer normalization
- ReLU activation to ensure non-negative outputs

## Data Structure

The MRDS dataset contains mineral distributions across the United States with qualities A-E, we represent them in five quality layers:
- **A layer**: Only A quality datapoints
- **B layer**: A and B quality datapoints
- **C layer**: A, B, and C quality datapoints
- **D layer**: A, B, C, and D quality datapoints
- **E layer**: All datapoints from A to E

Each file in the dataset directory represents a different mineral and contains these five quality layers.

## Training Procedure

1. **Data Preparation**:
   - Load the mineral data.
   - Reshape and process the data to fit the model input requirements.

2. **Model Training**:
   - The model is trained to minimize the Mean Squared Error (MSE) between predicted and actual mineral distributions.
   - A custom loss function is used to apply higher penalties for the A layer to avoid overprediction.
   - The total number of elements in the E layer is kept constant by normalizing the outputs during training.

3. **Evaluation and Metrics**:
   - After training, the model's performance is evaluated by comparing the predicted and actual distributions.
   - Metrics for each layer (A to E) are calculated and logged for analysis.

## Visualization

The results are visualized to compare predicted distributions with ground truth data. Dice coefficients are also computed for each layer to assess the model's accuracy.

## Key Functions and Methods

- **Model Architecture**:
  ```python

    grid_size = 60
    class SimplestMineralTransformer(nn.Module):
        def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
            super(SimplestMineralTransformer, self).__init__()
            self.input_projection = nn.Linear(grid_size * grid_size, d_model)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
            
            self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
            
            self.output_projection = nn.Linear(d_model, grid_size * grid_size)
            
            self.layer_norm = nn.LayerNorm(d_model)
            self.relu = nn.ReLU()  # Ensure non-negative output

        def forward(self, src, tgt):
            batch_size, seq_length_input, feature_dim_input = src.shape
            seq_length_output = tgt.shape[1]
            
            src = self.input_projection(src)
            tgt_proj = self.input_projection(tgt)
            
            src = self.layer_norm(src)
            tgt_proj = self.layer_norm(tgt_proj)
            
            memory = self.encoder(src)
            output = self.decoder(tgt_proj, memory)
            
            output = self.output_projection(output)
            output = self.relu(output)  # Apply ReLU to ensure non-negative output
            
            output = output.view(batch_size, seq_length_output, grid_size, grid_size)
            
            # Apply cumulative sum and then reshape
            output_split = torch.split(output, 5, dim=1)  # Split by quality layers (5 each)
            output_cumsum = []
            for layer in output_split:
                output_cumsum.append(torch.cumsum(layer, dim=1))
            
            output_cumsum = torch.cat(output_cumsum, dim=1)
            output_cumsum = output_cumsum.view(batch_size, 5, 5, grid_size, grid_size)
            
            return output_cumsum, output
