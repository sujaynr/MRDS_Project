import random
import torch
import torch.nn.functional as F
import pdb
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import copy

from utils import gaussian_filter





def mask_random_rectangle(array, mask_value=0):
    N, H, W = array.shape
    # Randomly choose rectangle dimensions
    rect_height = np.random.randint(1, H + 1)  # Random height (1 to H)
    rect_width = np.random.randint(1, W + 1)   # Random width (1 to W)
    
    # Randomly choose top-left corner of the rectangle
    start_y = np.random.randint(0, H - rect_height + 1)
    start_x = np.random.randint(0, W - rect_width + 1)
    
    # Apply the mask to all [N, :, :] slices
    array[:, start_y:start_y + rect_height, start_x:start_x + rect_width] = mask_value
    return array

class MineralDataset(Dataset): #FIX bug WITH INDEXING WHEN MINERALS NOT THERE
    def __init__(self, counts, input_minerals, output_mineral, indices, train=True, sigma=1, unet=False):
        self.counts = counts[indices]
        self.input_minerals = input_minerals
        self.output_mineral = output_mineral
        self.train = train
        self.sigma = sigma
        self.unet = unet

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        input_data = self.counts[idx].copy()
        input_data = (input_data > 0).astype(np.float32)


        if self.output_mineral is not None:
            output_data = copy.deepcopy(input_data[self.output_mineral:self.output_mineral+1, :, :])
            mask = np.zeros_like(output_data, dtype=np.uint8)
            input_data[self.output_mineral, :, :] = -1
            mask[self.output_mineral, :, :] = 1
        else:
            output_data = copy.deepcopy(input_data[0:10, :, :])
            mask = np.zeros_like(output_data, dtype=np.uint8)
            if np.random.uniform() < 0.5:
                input_data[0:10, :, :] = mask_random_rectangle(input_data[0:10, :, :], -1)
                mask[0:10, :, :] = 1 * (input_data[0:10, :, :] == -1)
            else:
                n = random.randint(1, 10)
                numbers = random.sample(range(10), n)
                mask[numbers, :, :] = 1
                input_data[numbers, :, :] = -1

        if self.unet:
            pad = (0, 14, 0, 14)  # Padding (left, right, top, bottom)
            input_data = np.pad(input_data, ((0, 0), pad[:2], pad[2:]), mode='constant', constant_values=0)
            output_data = np.pad(output_data, ((0, 0), pad[:2], pad[2:]), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, 0), pad[:2], pad[2:]), mode='constant', constant_values=0)


        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32), torch.tensor([idx], dtype=int), torch.tensor(mask, dtype=torch.float32)



grid_size = 50  # Adjusted to 50x50 grid



import torch
import torch.nn as nn

class SpatialTransformer(nn.Module):
    """
    A Transformer that interprets the spatial dimensions (H, W) as the sequence dimension.
    Input shape:  [B, N, H, W]
    Output shape: [B, M, H, W]
    
    Args:
        in_channels:  Number of input channels (N).
        out_channels: Number of output channels (M).
        hidden_dim:   Internal embedding dimension of the Transformer.
        nhead:        Number of attention heads.
        num_layers:   Number of Transformer encoder layers.
        max_size:     Maximum spatial size for the learnable embeddings
                      (must be >= max(H, W) you expect at runtime).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        max_size: int = 256,
    ):
        super().__init__()
        
        # Project from in_channels (N) to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Learnable row and column embeddings
        # We assume H, W <= max_size
        self.row_embed = nn.Embedding(max_size, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_size, hidden_dim // 2)
        
        # Define the Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            batch_first=False  # PyTorch Transformer expects (seq, batch, dim) by default
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Project from hidden_dim to out_channels (M)
        self.output_proj = nn.Linear(hidden_dim, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, H, W]
        
        Returns:
            [B, M, H, W]
        """
        b, n, h, w = x.shape
        
        # 1) Flatten spatial dimensions to a sequence: (H * W)
        #    -> shape: [B, N, H*W] -> [B, H*W, N]
        x = x.view(b, n, h * w).permute(0, 2, 1)  # [B, H*W, N]
        
        # 2) Project input to hidden_dim
        x = self.input_proj(x)  # [B, H*W, hidden_dim]
        
        # 3) Create 2D positional embeddings
        #    We'll embed row indices and column indices, then sum them.
        row_ids = torch.arange(h, device=x.device)
        col_ids = torch.arange(w, device=x.device)
        
        # shape: [h, hidden_dim//2], [w, hidden_dim//2]
        row_emb = self.row_embed(row_ids)  # [h, hidden_dim//2]
        col_emb = self.col_embed(col_ids)  # [w, hidden_dim//2]

        row_emb_expanded = row_emb.unsqueeze(1).expand(-1, w, -1)  # [h, w, hidden_dim//2]
        col_emb_expanded = col_emb.unsqueeze(0).expand(h, -1, -1)  # [h, w, hidden_dim//2]
        pos = torch.cat([row_emb_expanded, col_emb_expanded], dim=-1)  # [h, w, hidden_dim]
        
        # Flatten to [h*w, hidden_dim]
        pos = pos.view(h * w, -1)  # [h*w, hidden_dim]
        
        # Repeat pos for each element in the batch => [B, h*w, hidden_dim]
        pos = pos.unsqueeze(0).expand(b, -1, -1)  # [B, H*W, hidden_dim]

        # 4) Add positional embeddings
        x = x + pos  # [B, H*W, hidden_dim]
        
        # 5) The PyTorch Transformer expects input of shape [seq_len, batch_size, dim]
        x = x.permute(1, 0, 2)  # [H*W, B, hidden_dim]
        
        # 6) Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # [H*W, B, hidden_dim]
        
        # 7) Convert back to [B, H*W, hidden_dim]
        x = x.permute(1, 0, 2)  # [B, H*W, hidden_dim]
        
        # 8) Project to out_channels (M)
        x = self.output_proj(x)  # [B, H*W, M]
        
        # 9) Reshape back to [B, M, H, W]
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)  # [B, M, H, W]

        x = self.sigmoid(x)
        return x



class TransformerToConv(nn.Module): ############################### ARCH 2
    def __init__(self, input_dim, hidden_dim, intermediate_dim, d_model, nhead, num_layers, dropout_rate):
        super(TransformerToConv, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, d_model)
        self.d_model = d_model  # Save d_model as a class attribute
        self.hidden_dim = hidden_dim  # Save hidden_dim as a class attribute

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.fc4 = nn.Linear(d_model, hidden_dim * grid_size * grid_size)  # Fully connected layer to map back to the spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, src):
        batch_size = src.size(0)
        src_flat = src.view(batch_size, -1)
        # print(f"Input src shape: {src.shape}")
        x = self.relu(self.fc1(src_flat))
        # print(f"After fc1: {x.shape}")
        x = self.relu(self.fc2(x))
        # print(f"After fc2: {x.shape}")
        x = self.relu(self.fc3(x))
        # print(f"After fc3: {x.shape}")

        x = x.view(batch_size, -1, self.d_model)  # Change shape to (batch_size, seq_len, d_model)
        # print(f"After reshaping: {x.shape}")
        x = self.transformer_encoder(x)
        # print(f"After transformer encoder: {x.shape}")

        x = self.relu(self.fc4(x))
        # print(f"After fc4: {x.shape}")
        x = x.view(batch_size, self.hidden_dim, grid_size, grid_size)  # Reshape to (batch_size, hidden_dim, grid_size, grid_size)
        # print(f"After reshaping to spatial dimensions: {x.shape}")

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x



class LinToConv(nn.Module): ############################### ARCH 1
    def __init__(self, input_dim, hidden_dim, intermediate_dim):
        super(LinToConv, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, grid_size * grid_size)
        
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, src):
        batch_size = src.size(0)
        src_flat = src.view(batch_size, -1)

        x = self.relu(self.fc1(src_flat))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = x.view(batch_size, 1, grid_size, grid_size)

        x = self.relu(self.conv1(src))
        x = self.maxpool(x)
        x = self.upsample(x)
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module): ############################### ARCH 3
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.unet = smp.Unet(
            encoder_name='resnet34', 
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.unet(x)
        x = self.sigmoid(x)
        return x




















































class MineralTransformer(nn.Module): 
    def __init__(self, d_model=512, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024, dropout=0.5):
        super(MineralTransformer, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the output size after the conv and pool layers
        conv_output_size = grid_size // 4  # After two pooling layers with kernel size 2

        # Linear layer to project the CNN output to the transformer input size
        self.input_projection = nn.Linear(conv_output_size * conv_output_size * d_model, d_model)
        self.target_projection = nn.Linear(grid_size * grid_size, d_model)

        # Transformer encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        # Linear layer to project the transformer output to the original grid size
        self.output_projection = nn.Linear(d_model, grid_size * grid_size)

        # Layer normalization and ReLU activation
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()  # Ensure non-negative output

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        '''
        Input src shape: torch.Size([16, 10, 50, 50])
        Input tgt shape: torch.Size([16, 1, 50, 50])
        After conv1 and pool: torch.Size([16, 32, 25, 25])
        After conv2 and pool: torch.Size([16, 64, 12, 12])
        After conv3: torch.Size([16, 256, 12, 12])
        After conv4: torch.Size([16, 512, 12, 12])
        After flattening: torch.Size([16, 73728])
        After input projection: torch.Size([16, 512])
        After target projection: torch.Size([16, 512])
        After layer norm on src: torch.Size([16, 512])
        After layer norm on tgt_proj: torch.Size([16, 512])
        After transformer encoder (memory): torch.Size([16, 1, 512])
        After transformer decoder: torch.Size([16, 1, 512])
        After output projection: torch.Size([16, 2500])
        After ReLU: torch.Size([16, 2500])
        Final output shape: torch.Size([16, 1, 50, 50])
        '''

        batch_size, num_minerals_input, grid_x, grid_y = src.shape
        num_minerals_output = tgt.shape[1]

        x = self.pool(self.relu(self.conv1(src)))  # First convolutional layer with ReLU and pooling
        x = self.pool(self.relu(self.conv2(x)))    # Second convolutional layer with ReLU and pooling
        x = self.relu(self.conv3(x))               # Third convolutional layer with ReLU
        x = self.relu(self.conv4(x))               # Fourth convolutional layer with ReLU

        # Flatten the CNN output and project it to the transformer input size
        x = x.view(batch_size, -1)
        src = self.input_projection(x)

        # Process the target data for the transformer decoder
        tgt_proj = self.target_projection(tgt.view(batch_size, -1))

        # Apply layer normalization
        src = self.layer_norm(src)
        tgt_proj = self.layer_norm(tgt_proj)

        # Pass the projected input through the transformer encoder
        memory = self.encoder(src.unsqueeze(1), src_key_padding_mask=src_key_padding_mask)

        # Pass the projected target and encoder output through the transformer decoder
        output = self.decoder(tgt_proj.unsqueeze(1), memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        # Project the transformer output back to the original grid size and apply ReLU
        output = self.output_projection(output.squeeze(1))
        output = self.relu(output)  # Ensure non-negative output

        # Reshape the output to match the original grid size
        output = output.view(batch_size, num_minerals_output, grid_x, grid_y)

        return output

grid_size = 50  # Adjusted to 50x50 grid

class SimplifiedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimplifiedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, grid_size * grid_size)
        self.relu = nn.ReLU()

    def forward(self, src, tgt):
        batch_size = src.size(0)
        src_flat = src.view(batch_size, -1)
        tgt_flat = tgt.view(batch_size, -1)

        x = self.relu(self.fc1(src_flat))
        output = self.fc2(x)
        output = self.relu(output)

        output = output.view(batch_size, 1, grid_size, grid_size)

        return output


class LinToTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, intermediate_dim, d_model, nhead, num_layers, dropout_rate):
        super(LinToTransformer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, d_model)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        
        self.fc4 = nn.Linear(d_model, grid_size * grid_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, src):
        batch_size = src.size(0)
        src_flat = src.view(batch_size, -1)

        x = self.relu(self.fc1(src_flat))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))

        x = x.view(batch_size, 1, -1)  # Change shape to (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)
        x = x.view(batch_size, -1)  # Flatten the output

        x = self.relu(self.fc4(x))
        x = x.view(batch_size, 1, grid_size, grid_size)  # Reshape back to (batch_size, 1, grid_size, grid_size)

        return x

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
