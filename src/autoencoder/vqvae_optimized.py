"""
Optimized VQ-VAE architecture for natural Terraria world generation.

Based on analysis of 463M tiles from 23 worlds, this version:
- Uses only 218 naturally-occurring block types (vs 693 total)
- Uses only 77 naturally-occurring wall types (vs 347 total)
- Removes non-natural features: paint, illuminants, echo coating, block_active
- Simplifies liquids to binary present/absent + type
- Reduces from 17 to 9 channels (47% reduction)
- Saves 745 embedding table entries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .quantizer import VectorQuantizer, VectorQuantizerEMA
from .encoder import Encoder
from .decoder import Decoder
from ..terraria.natural_ids import (
    NATURAL_BLOCK_IDS, 
    NATURAL_WALL_IDS,
    BLOCK_ID_TO_INDEX,
    WALL_ID_TO_INDEX, 
    BLOCK_INDEX_TO_ID,
    WALL_INDEX_TO_ID
)

# Channel indices for the 9-channel optimized tensor
CHANNEL_BLOCK_TYPE = 0
CHANNEL_BLOCK_SHAPE = 1
CHANNEL_WALL_TYPE = 2
CHANNEL_LIQUID_PRESENT = 3
CHANNEL_LIQUID_TYPE = 4
CHANNEL_WIRE_RED = 5
CHANNEL_WIRE_BLUE = 6
CHANNEL_WIRE_GREEN = 7
CHANNEL_ACTUATOR = 8

NUM_NATURAL_BLOCKS = len(NATURAL_BLOCK_IDS)  # 218
NUM_NATURAL_WALLS = len(NATURAL_WALL_IDS)    # 77
NUM_LIQUID_TYPES = 5  # 0=none, 1=water, 2=lava, 3=honey, 4=shimmer


class OptimizedTileEncoder(nn.Module):
    """Encoder that embeds categorical features for natural tiles."""
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Embeddings for categorical features (using natural IDs only)
        self.block_embedding = nn.Embedding(NUM_NATURAL_BLOCKS, embedding_dim)
        self.wall_embedding = nn.Embedding(NUM_NATURAL_WALLS, embedding_dim)
        self.liquid_embedding = nn.Embedding(NUM_LIQUID_TYPES, embedding_dim)
        
        # Total continuous channels: block_shape(1) + liquid_present(1) + wires(3) + actuator(1) = 6
        self.num_continuous = 6
        self.num_categorical = 3  # block, wall, liquid_type
        
        # Output will be concatenated embeddings + continuous features
        self.output_dim = (self.num_categorical * embedding_dim) + self.num_continuous
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, H, W) tensor with channels as defined above
            
        Returns:
            (B, output_dim, H, W) tensor with embedded categorical features
        """
        B, C, H, W = x.shape
        assert C == 9, f"Expected 9 channels, got {C}"
        
        # Extract channels
        block_type = x[:, CHANNEL_BLOCK_TYPE, :, :].long()  # (B, H, W)
        block_shape = x[:, CHANNEL_BLOCK_SHAPE, :, :]  # (B, H, W)
        wall_type = x[:, CHANNEL_WALL_TYPE, :, :].long()  # (B, H, W)
        liquid_present = x[:, CHANNEL_LIQUID_PRESENT, :, :]  # (B, H, W)
        liquid_type = x[:, CHANNEL_LIQUID_TYPE, :, :].long()  # (B, H, W)
        wire_red = x[:, CHANNEL_WIRE_RED, :, :]  # (B, H, W)
        wire_blue = x[:, CHANNEL_WIRE_BLUE, :, :]  # (B, H, W)
        wire_green = x[:, CHANNEL_WIRE_GREEN, :, :]  # (B, H, W)
        actuator = x[:, CHANNEL_ACTUATOR, :, :]  # (B, H, W)
        
        # Clamp categorical indices to valid ranges
        block_type = torch.clamp(block_type, 0, NUM_NATURAL_BLOCKS - 1)
        wall_type = torch.clamp(wall_type, 0, NUM_NATURAL_WALLS - 1)
        liquid_type = torch.clamp(liquid_type, 0, NUM_LIQUID_TYPES - 1)
        
        # Embed categorical features
        # Reshape to (B*H*W,) for embedding, then back to (B, H, W, embed_dim)
        block_emb = self.block_embedding(block_type.view(-1)).view(B, H, W, self.embedding_dim)
        wall_emb = self.wall_embedding(wall_type.view(-1)).view(B, H, W, self.embedding_dim)
        liquid_emb = self.liquid_embedding(liquid_type.view(-1)).view(B, H, W, self.embedding_dim)
        
        # Permute embeddings to (B, embed_dim, H, W)
        block_emb = block_emb.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)
        wall_emb = wall_emb.permute(0, 3, 1, 2)
        liquid_emb = liquid_emb.permute(0, 3, 1, 2)
        
        # Prepare continuous features (add channel dimension)
        continuous = torch.stack([
            block_shape,
            liquid_present,
            wire_red,
            wire_blue,
            wire_green,
            actuator
        ], dim=1)  # (B, 6, H, W)
        
        # Concatenate all features
        encoded = torch.cat([
            block_emb,      # (B, embed_dim, H, W)
            wall_emb,       # (B, embed_dim, H, W)
            liquid_emb,     # (B, embed_dim, H, W)
            continuous      # (B, 6, H, W)
        ], dim=1)  # (B, 3*embed_dim + 6, H, W)
        
        return encoded


class OptimizedTileDecoder(nn.Module):
    """Decoder that outputs both classification and continuous predictions."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # Classification heads for categorical features
        self.block_classifier = nn.Conv2d(input_dim, NUM_NATURAL_BLOCKS, 1)
        self.wall_classifier = nn.Conv2d(input_dim, NUM_NATURAL_WALLS, 1)
        self.liquid_classifier = nn.Conv2d(input_dim, NUM_LIQUID_TYPES, 1)
        
        # Regression head for continuous features (6 channels)
        self.continuous_head = nn.Conv2d(input_dim, 6, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, input_dim, H, W) latent features
            
        Returns:
            Tuple of:
            - reconstructed: (B, 9, H, W) reconstructed tile tensor
            - logits: Tuple of (block_logits, wall_logits, liquid_logits)
        """
        # Get logits for categorical features
        block_logits = self.block_classifier(x)  # (B, 218, H, W)
        wall_logits = self.wall_classifier(x)    # (B, 77, H, W)
        liquid_logits = self.liquid_classifier(x)  # (B, 5, H, W)
        
        # Get continuous predictions
        continuous = torch.sigmoid(self.continuous_head(x))  # (B, 6, H, W)
        
        # For categorical channels, use argmax to get predicted class
        block_pred = torch.argmax(block_logits, dim=1, keepdim=True).float()  # (B, 1, H, W)
        wall_pred = torch.argmax(wall_logits, dim=1, keepdim=True).float()
        liquid_pred = torch.argmax(liquid_logits, dim=1, keepdim=True).float()
        
        # Reconstruct 9-channel tensor
        reconstructed = torch.cat([
            block_pred,                      # Channel 0: block type (index)
            continuous[:, 0:1, :, :],        # Channel 1: block shape
            wall_pred,                       # Channel 2: wall type (index)
            continuous[:, 1:2, :, :],        # Channel 3: liquid present
            liquid_pred,                     # Channel 4: liquid type
            continuous[:, 2:3, :, :],        # Channel 5: wire red
            continuous[:, 3:4, :, :],        # Channel 6: wire blue
            continuous[:, 4:5, :, :],        # Channel 7: wire green
            continuous[:, 5:6, :, :],        # Channel 8: actuator
        ], dim=1)  # (B, 9, H, W)
        
        return reconstructed, (block_logits, wall_logits, liquid_logits)


class VQVAEOptimized(nn.Module):
    """Optimized VQ-VAE for natural Terraria terrain generation."""
    
    def __init__(
        self,
        embedding_dim: int = 32,
        h_dim: int = 128,
        res_h_dim: int = 64,
        n_embeddings: int = 512,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_reset_threshold: float = 0.5,
        ema_reset_interval: int = 500,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_ema = use_ema
        self.n_embeddings = n_embeddings
        
        # Tile-specific encoder/decoder
        self.tile_encoder = OptimizedTileEncoder(embedding_dim)
        encoder_output_dim = self.tile_encoder.output_dim  # 3*32 + 6 = 102
        
        # Standard convolutional encoder
        self.conv_encoder = Encoder(encoder_output_dim, h_dim, 2, res_h_dim)
        
        # Vector quantizer - use EMA version by default to prevent codebook collapse
        if use_ema:
            self.vq = VectorQuantizerEMA(
                n_e=n_embeddings,
                e_dim=h_dim,
                beta=beta,
                decay=ema_decay,
                reset_threshold=ema_reset_threshold,
                reset_interval=ema_reset_interval,
            )
        else:
            self.vq = VectorQuantizer(n_embeddings, h_dim, beta)
        
        # Standard convolutional decoder  
        self.conv_decoder = Decoder(h_dim, res_h_dim, 2, res_h_dim, out_dim=res_h_dim)
        
        # Tile-specific decoder
        self.tile_decoder = OptimizedTileDecoder(res_h_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, 9, H, W) input tile tensor
            
        Returns:
            embedding_loss: VQ loss
            x_hat: (B, 9, H, W) reconstructed tiles
            perplexity: Codebook usage metric
            logits: Tuple of (block_logits, wall_logits, liquid_logits) for loss computation
        """
        # Encode with tile-specific embeddings
        tile_encoded = self.tile_encoder(x)  # (B, 102, H, W)
        
        # Convolutional encoding
        z_e = self.conv_encoder(tile_encoded)  # (B, h_dim, H', W')
        
        # Vector quantization
        embedding_loss, z_q, perplexity, _, _ = self.vq(z_e)
        
        # Convolutional decoding
        decoded = self.conv_decoder(z_q)  # (B, res_h_dim, H', W')
        
        # Tile-specific decoding
        x_hat, logits = self.tile_decoder(decoded)
        
        return embedding_loss, x_hat, perplexity, logits
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct without computing losses."""
        _, x_hat, _, _ = self.forward(x)
        return x_hat
    
    def get_codebook_usage_stats(self) -> Optional[dict]:
        """
        Get detailed codebook usage statistics.
        
        Only available when using EMA quantizer.
        
        Returns:
            Dictionary with usage stats, or None if not using EMA.
        """
        if self.use_ema and hasattr(self.vq, 'get_codebook_usage_stats'):
            stats = self.vq.get_codebook_usage_stats()
            stats['codebook_size'] = self.n_embeddings
            stats['usage_percent'] = stats['active_codes'] / self.n_embeddings * 100
            return stats
        return None


def compute_optimized_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    embedding_loss: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Compute loss for optimized model.
    
    Args:
        x: Original input (B, 9, H, W)
        x_hat: Reconstruction (B, 9, H, W)
        logits: Tuple of (block_logits, wall_logits, liquid_logits)
        embedding_loss: VQ embedding loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    block_logits, wall_logits, liquid_logits = logits
    
    # Extract ground truth
    block_target = x[:, CHANNEL_BLOCK_TYPE, :, :].long()
    wall_target = x[:, CHANNEL_WALL_TYPE, :, :].long()
    liquid_target = x[:, CHANNEL_LIQUID_TYPE, :, :].long()
    
    # Clamp targets
    block_target = torch.clamp(block_target, 0, NUM_NATURAL_BLOCKS - 1)
    wall_target = torch.clamp(wall_target, 0, NUM_NATURAL_WALLS - 1)
    liquid_target = torch.clamp(liquid_target, 0, NUM_LIQUID_TYPES - 1)
    
    # Categorical losses (cross-entropy)
    block_loss = F.cross_entropy(block_logits, block_target)
    wall_loss = F.cross_entropy(wall_logits, wall_target)
    liquid_loss = F.cross_entropy(liquid_logits, liquid_target)
    
    # Continuous losses (MSE) for channels 1,3,5,6,7,8
    continuous_indices = [CHANNEL_BLOCK_SHAPE, CHANNEL_LIQUID_PRESENT, 
                         CHANNEL_WIRE_RED, CHANNEL_WIRE_BLUE, 
                         CHANNEL_WIRE_GREEN, CHANNEL_ACTUATOR]
    continuous_loss = F.mse_loss(
        x_hat[:, continuous_indices, :, :],
        x[:, continuous_indices, :, :]
    )
    
    # Weighted combination
    categorical_loss = block_loss + wall_loss + liquid_loss
    reconstruction_loss = categorical_loss + continuous_loss
    total_loss = reconstruction_loss + embedding_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'reconstruction': reconstruction_loss.item(),
        'categorical': categorical_loss.item(),
        'block': block_loss.item(),
        'wall': wall_loss.item(),
        'liquid': liquid_loss.item(),
        'continuous': continuous_loss.item(),
        'embedding': embedding_loss.item()
    }
    
    return total_loss, loss_dict
