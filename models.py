import torch
import torch.nn as nn
import pdb

grid_size = 50  # Adjusted to 50x50 grid

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
        batch_size, num_minerals_input, grid_x, grid_y = src.shape
        # torch.Size([16, 9, 50, 50])
        num_minerals_output = tgt.shape[1]
        # torch.Size([16, 1, 50, 50])
        # pdb.set_trace()
        src = src.view(batch_size, num_minerals_input, -1)
        # torch.Size([16, 9, 2500])
        tgt = tgt.view(batch_size, num_minerals_output, -1)
        # torch.Size([16, 1, 2500])
        
        src = self.input_projection(src)
        # (torch.Size([16, 9, 256])
        tgt_proj = self.input_projection(tgt)
        # (torch.Size([16, 1, 256])
        
        src = self.layer_norm(src)
        # (torch.Size([16, 9, 256])
        tgt_proj = self.layer_norm(tgt_proj)
        # (torch.Size([16, 1, 256])
        
        memory = self.encoder(src)
        # torch.Size([16, 9, 256])
        output = self.decoder(tgt_proj, memory)
        # torch.Size([16, 1, 256])
        
        output = self.output_projection(output)
        # torch.Size([16, 1, 2500])
        output = self.relu(output)  # Apply ReLU to ensure non-negative output
        # torch.Size([16, 1, 2500])
        
        output = output.view(batch_size, num_minerals_output, grid_x, grid_y)
        # torch.Size([16, 1, 50, 50])
        
        return output


grid_sizeS = 60

class SimpleMineralTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
        super(SimpleMineralTransformer, self).__init__()
        self.input_projection = nn.Linear(grid_sizeS * grid_sizeS, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, grid_sizeS * grid_sizeS)
        
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
        
        output = output.view(batch_size, seq_length_output, grid_sizeS, grid_sizeS)
        
        # Apply cumulative sum and then reshape
        output_split = torch.split(output, 5, dim=1)  # Split by quality layers (5 each)
        output_cumsum = []
        for layer in output_split:
            output_cumsum.append(torch.cumsum(layer, dim=1))
        
        output_cumsum = torch.cat(output_cumsum, dim=1)
        output_cumsum = output_cumsum.view(batch_size, 5, 5, grid_sizeS, grid_sizeS)
        
        return output_cumsum, output
