# models.py

import torch
import torch.nn as nn

grid_size = 30

class MineralTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(MineralTransformer, self).__init__()
        self.input_projection = nn.Linear(grid_size * grid_size, d_model)
        self.output_projection = nn.Linear(grid_size * grid_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), 
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout), 
            num_decoder_layers
        )
        self.final_projection = nn.Linear(d_model, grid_size * grid_size)
        
    def forward(self, src, tgt):
        batch_size, seq_length, feature_dim = src.shape
        
        src = src.view(batch_size * seq_length, feature_dim)
        src = self.input_projection(src).view(batch_size, seq_length, -1)
        
        tgt = tgt.view(batch_size * seq_length, feature_dim)
        tgt = self.output_projection(tgt).view(batch_size, seq_length, -1)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        output = self.final_projection(output.view(batch_size * seq_length, -1))
        return output.view(batch_size, seq_length, grid_size, grid_size)
