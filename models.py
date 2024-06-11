# import torch
# import torch.nn as nn

# grid_size = 30

# class MineralTransformer(nn.Module):
#     def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
#         super(MineralTransformer, self).__init__()
#         self.input_projection = nn.Linear(grid_size * grid_size, d_model)
#         self.output_projection = nn.Linear(grid_size * grid_size, d_model)
#         self.encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), 
#             num_encoder_layers
#         )
#         self.decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout), 
#             num_decoder_layers
#         )
#         self.final_projection = nn.Linear(d_model, grid_size * grid_size)
        
#     def forward(self, src, tgt):
#         batch_size, seq_length, feature_dim = src.shape
        
#         src = src.view(batch_size * seq_length, feature_dim)
#         src = self.input_projection(src).view(batch_size, seq_length, -1)
        
#         tgt = tgt.view(batch_size * seq_length, feature_dim)
#         tgt = self.output_projection(tgt).view(batch_size, seq_length, -1)
        
#         memory = self.encoder(src)
#         output = self.decoder(tgt, memory)
        
#         output = self.final_projection(output.view(batch_size * seq_length, -1))
#         return output.view(batch_size, seq_length, grid_size, grid_size)
import torch
import torch.nn as nn

grid_size = 30

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
        batch_size, seq_length, feature_dim = src.shape
        
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        output = self.output_projection(output)
        return output.view(batch_size, seq_length, grid_size, grid_size)