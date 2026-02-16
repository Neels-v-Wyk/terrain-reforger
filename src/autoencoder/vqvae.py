"""
VQ-VAE with embedding layers for categorical features.

This version treats categorical features (block types, wall types) as proper
categories rather than continuous values, which is more appropriate for the data.
"""

import torch
import torch.nn as nn
import lihzahrd.enums as enums
from .encoder import Encoder
from .quantizer import VectorQuantizer
from .decoder import Decoder


# Get actual maximum values from lihzahrd
BLOCK_TYPE_COUNT = len(list(enums.BlockType)) + 1  # +1 for "no block" (0)
WALL_TYPE_COUNT = len(list(enums.WallType)) + 1    # +1 for "no wall" (0)
LIQUID_TYPE_COUNT = len(list(enums.LiquidType)) + 1  # Includes "no liquid"
SHAPE_COUNT = 6  # 0-5: normal, half, 4 slopes


class TileCategoricalEncoder(nn.Module):
    """
    Encodes tile data with embeddings for categorical features.
    
    Architecture:
    1. Embed categorical features (block_type, wall_type, etc.)
    2. Concatenate with continuous features (liquid_amount, binary flags)
    3. Pass through convolutional encoder
    """
    
    def __init__(self, embedding_dim: int = 16, h_dim: int = 128, 
                 n_res_layers: int = 3, res_h_dim: int = 64):
        super().__init__()
        
        # Embedding layers for categorical features
        self.block_type_embed = nn.Embedding(BLOCK_TYPE_COUNT, embedding_dim)
        self.wall_type_embed = nn.Embedding(WALL_TYPE_COUNT, embedding_dim)
        self.shape_embed = nn.Embedding(SHAPE_COUNT, embedding_dim // 2)
        self.liquid_type_embed = nn.Embedding(LIQUID_TYPE_COUNT, embedding_dim // 2)
        
        # Paint embeddings (smaller since only 32 colors)
        self.block_paint_embed = nn.Embedding(32, embedding_dim // 4)
        self.wall_paint_embed = nn.Embedding(32, embedding_dim // 4)
        
        # Calculate total input channels after embedding
        # Embedded: block_type(16) + wall_type(16) + shape(8) + liquid_type(8) + 
        #           block_paint(4) + wall_paint(4) = 56
        # Continuous: block_active(1) + block_illum(1) + block_echo(1) +
        #            wall_illum(1) + wall_echo(1) + liquid_amount(1) +
        #            4 wires(4) + actuator(1) = 11
        # Total: 56 + 11 = 67 channels
        total_channels = (embedding_dim * 2 +  # block_type + wall_type
                         embedding_dim // 2 * 2 +  # shape + liquid_type
                         embedding_dim // 4 * 2 +  # block_paint + wall_paint
                         11)  # continuous features
        
        # Convolutional encoder (same as before but with more input channels)
        self.encoder = Encoder(total_channels, h_dim, n_res_layers, res_h_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 17, H, W) with raw channel values
            
        Returns:
            Encoded tensor (B, h_dim, H', W')
        """
        B, C, H, W = x.shape
        
        # Extract and embed categorical features
        # Convert to long for embedding lookup
        block_type = self.block_type_embed(x[:, 0].long())  # (B, H, W, emb_dim)
        wall_type = self.wall_type_embed(x[:, 6].long())
        shape = self.shape_embed(x[:, 2].long())
        liquid_type = self.liquid_type_embed(x[:, 10].long())
        block_paint = self.block_paint_embed(x[:, 3].long())
        wall_paint = self.wall_paint_embed(x[:, 7].long())
        
        # Rearrange embeddings: (B, H, W, E) -> (B, E, H, W)
        block_type = block_type.permute(0, 3, 1, 2)
        wall_type = wall_type.permute(0, 3, 1, 2)
        shape = shape.permute(0, 3, 1, 2)
        liquid_type = liquid_type.permute(0, 3, 1, 2)
        block_paint = block_paint.permute(0, 3, 1, 2)
        wall_paint = wall_paint.permute(0, 3, 1, 2)
        
        # Extract continuous features (already in correct format)
        continuous = torch.cat([
            x[:, 1:2],   # block_active
            x[:, 4:6],   # block_illuminant, block_echo
            x[:, 8:10],  # wall_illuminant, wall_echo
            x[:, 11:12], # liquid_amount
            x[:, 12:17], # wires + actuator
        ], dim=1)  # (B, 11, H, W)
        
        # Concatenate all features
        combined = torch.cat([
            block_type, wall_type, shape, liquid_type,
            block_paint, wall_paint, continuous
        ], dim=1)  # (B, 67, H, W)
        
        # Pass through convolutional encoder
        return self.encoder(combined)


class TileCategoricalDecoder(nn.Module):
    """
    Decodes latent representation back to tile data with classification heads
    for categorical features.
    """
    
    def __init__(self, embedding_dim: int = 64, h_dim: int = 128,
                 n_res_layers: int = 3, res_h_dim: int = 64):
        super().__init__()
        
        # Base decoder to get spatial features
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        
        # Classification heads for categorical features
        # Note: Decoder outputs h_dim channels, we need to project to logits
        
        # Get output channels from decoder (it outputs to channel 17 in original)
        # We'll modify it to output h_dim channels instead
        self.feature_projection = nn.Conv2d(17, h_dim, kernel_size=1)
        
        # Separate classification heads
        self.block_type_head = nn.Conv2d(h_dim, BLOCK_TYPE_COUNT, kernel_size=1)
        self.wall_type_head = nn.Conv2d(h_dim, WALL_TYPE_COUNT, kernel_size=1)
        self.shape_head = nn.Conv2d(h_dim, SHAPE_COUNT, kernel_size=1)
        self.liquid_type_head = nn.Conv2d(h_dim, LIQUID_TYPE_COUNT, kernel_size=1)
        self.block_paint_head = nn.Conv2d(h_dim, 32, kernel_size=1)
        self.wall_paint_head = nn.Conv2d(h_dim, 32, kernel_size=1)
        
        # Regression head for continuous features (11 channels)
        self.continuous_head = nn.Conv2d(h_dim, 11, kernel_size=1)
    
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: Latent tensor (B, embedding_dim, H', W')
            
        Returns:
            Tuple of (categorical_logits, continuous_values)
            categorical_logits: dict with keys for each categorical feature
            continuous_values: tensor (B, 11, H, W)
        """
        # Decode to spatial features
        features = self.decoder(z)  # (B, 17, H, W) - from original decoder
        features = self.feature_projection(features)  # (B, h_dim, H, W)
        
        # Get categorical logits
        categorical_logits = {
            'block_type': self.block_type_head(features),
            'wall_type': self.wall_type_head(features),
            'shape': self.shape_head(features),
            'liquid_type': self.liquid_type_head(features),
            'block_paint': self.block_paint_head(features),
            'wall_paint': self.wall_paint_head(features),
        }
        
        # Get continuous values
        continuous = torch.sigmoid(self.continuous_head(features))
        
        return categorical_logits, continuous


class VQVAEEmbedded(nn.Module):
    """
    VQ-VAE with proper categorical embeddings for tile features.
    
    This version:
    - Uses nn.Embedding for block types, wall types, etc.
    - Classification loss for categorical features
    - MSE loss for continuous features
    - Better handles the discrete nature of tile data
    """
    
    def __init__(self, h_dim=128, res_h_dim=64, n_res_layers=3,
                 n_embeddings=512, embedding_dim=64, beta=0.25,
                 categorical_embedding_dim=16):
        super().__init__()
        
        self.encoder = TileCategoricalEncoder(
            embedding_dim=categorical_embedding_dim,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim
        )
        
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        
        self.decoder = TileCategoricalDecoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim
        )
    
    def forward(self, x, return_indices=False):
        """
        Args:
            x: Input tensor (B, 17, H, W) - NOT normalized, raw values
            
        Returns:
            embedding_loss, reconstructed_output, perplexity [, indices]
        """
        # Encode
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        
        # Quantize
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = \
            self.vector_quantization(z_e)
        
        # Decode
        categorical_logits, continuous_pred = self.decoder(z_q)
        
        if return_indices:
            return embedding_loss, (categorical_logits, continuous_pred), \
                   perplexity, min_encoding_indices
        return embedding_loss, (categorical_logits, continuous_pred), perplexity
    
    def reconstruct_tensor(self, categorical_logits, continuous_pred):
        """
        Convert logits and continuous predictions back to standard (B, 17, H, W) format.
        
        Args:
            categorical_logits: Dict of logit tensors
            continuous_pred: Tensor (B, 11, H, W)
            
        Returns:
            Reconstructed tensor (B, 17, H, W) matching input format
        """
        # Get predictions from logits (argmax for categories)
        block_type = torch.argmax(categorical_logits['block_type'], dim=1, keepdim=True).float()
        wall_type = torch.argmax(categorical_logits['wall_type'], dim=1, keepdim=True).float()
        shape = torch.argmax(categorical_logits['shape'], dim=1, keepdim=True).float()
        liquid_type = torch.argmax(categorical_logits['liquid_type'], dim=1, keepdim=True).float()
        block_paint = torch.argmax(categorical_logits['block_paint'], dim=1, keepdim=True).float()
        wall_paint = torch.argmax(categorical_logits['wall_paint'], dim=1, keepdim=True).float()
        
        # Reconstruct full tensor
        reconstructed = torch.cat([
            block_type,          # 0
            continuous_pred[:, 0:1],  # 1: block_active
            shape,               # 2
            block_paint,         # 3
            continuous_pred[:, 1:3],  # 4-5: block_illuminant, block_echo
            wall_type,           # 6
            wall_paint,          # 7
            continuous_pred[:, 3:5],  # 8-9: wall_illuminant, wall_echo
            liquid_type,         # 10
            continuous_pred[:, 5:6],  # 11: liquid_amount
            continuous_pred[:, 6:11], # 12-16: wires + actuator
        ], dim=1)
        
        return reconstructed


def compute_categorical_loss(categorical_logits, x_target):
    """
    Compute cross-entropy loss for categorical features.
    
    Args:
        categorical_logits: Dict of logit tensors from decoder
        x_target: Target tensor (B, 17, H, W) with ground truth
        
    Returns:
        Total categorical loss (scalar)
    """
    loss = 0.0
    
    # Block type loss (channel 0)
    loss += nn.functional.cross_entropy(
        categorical_logits['block_type'],
        x_target[:, 0].long(),
        ignore_index=0  # Optionally ignore "no block"
    )
    
    # Wall type loss (channel 6)
    loss += nn.functional.cross_entropy(
        categorical_logits['wall_type'],
        x_target[:, 6].long(),
        ignore_index=0
    )
    
    # Shape loss (channel 2)
    loss += nn.functional.cross_entropy(
        categorical_logits['shape'],
        x_target[:, 2].long()
    )
    
    # Liquid type loss (channel 10)
    loss += nn.functional.cross_entropy(
        categorical_logits['liquid_type'],
        x_target[:, 10].long()
    )
    
    # Paint losses (channels 3, 7)
    loss += 0.1 * nn.functional.cross_entropy(  # Lower weight for paint
        categorical_logits['block_paint'],
        x_target[:, 3].long()
    )
    loss += 0.1 * nn.functional.cross_entropy(
        categorical_logits['wall_paint'],
        x_target[:, 7].long()
    )
    
    return loss


def compute_continuous_loss(continuous_pred, x_target):
    """
    Compute MSE loss for continuous features.
    
    Args:
        continuous_pred: Predicted continuous values (B, 11, H, W)
        x_target: Target tensor (B, 17, H, W)
        
    Returns:
        MSE loss (scalar)
    """
    # Extract continuous target channels
    continuous_target = torch.cat([
        x_target[:, 1:2],   # block_active
        x_target[:, 4:6],   # block_illuminant, block_echo
        x_target[:, 8:10],  # wall_illuminant, wall_echo
        x_target[:, 11:12], # liquid_amount
        x_target[:, 12:17], # wires + actuator
    ], dim=1)
    
    return nn.functional.mse_loss(continuous_pred, continuous_target)
