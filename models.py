import torch
import torch.nn as nn
import pdb

grid_size = 60

class MineralTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
        super(MineralTransformer, self).__init__()
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
