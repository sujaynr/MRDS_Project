import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

grid_size = 60

class MineralTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2):
        super(MineralTransformer, self).__init__()
        self.input_projection = nn.Linear(grid_size * grid_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, grid_size * grid_size)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        batch_size, seq_length_input, feature_dim_input = src.shape
        seq_length_output = tgt.shape[1]  # Get the sequence length of the output
        # print("src shape:", src.shape, "tgt shape:", tgt.shape)
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        # print("src shape:", src.shape, "tgt shape:", tgt.shape)
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        # print("src shape:", src.shape, "tgt shape:", tgt.shape)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        # print("output shape:", output.shape)
        output = self.output_projection(output)
        # print("output shape:", output.shape)
        
        output = output.view(batch_size, seq_length_output, grid_size, grid_size)

        output_cumsum = torch.cumsum(output, dim=1)
        
        return output_cumsum
