"""
Data normalization utilities for Terraria tile data.

This module handles normalization and denormalization of tile data
to make it suitable for neural network training.
"""

import torch
import numpy as np
from typing import Tuple
import lihzahrd.enums as enums


# Query lihzahrd for actual maximum values (cached at module load)
_BLOCK_TYPE_MAX = max([int(t.value) for t in enums.BlockType]) + 1  # +1 for buffer
_WALL_TYPE_MAX = max([int(t.value) for t in enums.WallType]) + 1
_LIQUID_TYPE_MAX = max([int(t.value) for t in enums.LiquidType])

BLOCK_TYPE_MAX = float(_BLOCK_TYPE_MAX)
WALL_TYPE_MAX = float(_WALL_TYPE_MAX)
PAINT_MAX = 31.0
LIQUID_TYPE_MAX = float(_LIQUID_TYPE_MAX)


class TileNormalizer:
    """
    Normalizes and denormalizes Terraria tile tensor data.
    
    Channel mapping (17 channels):
        0: Block type (0-700) -> normalized to [0, 1]
        1: Block active (0-1) -> already normalized
        2: Block shape (0-5) -> normalized to [0, 1]
        3: Block paint (0-31) -> normalized to [0, 1]
        4-5: Block illuminant/echo (0-1) -> already normalized
        6: Wall type (0-350) -> normalized to [0, 1]
        7: Wall paint (0-31) -> normalized to [0, 1]
        8-9: Wall illuminant/echo (0-1) -> already normalized
        10: Liquid type (0-3) -> normalized to [0, 1]
        11: Liquid amount (0-1) -> already normalized
        12-16: Wiring (0-1) -> already normalized
    """
    
    def __init__(self):
        # Define max values for each channel
        self.channel_maxes = torch.tensor([
            BLOCK_TYPE_MAX,    # 0: Block type
            1.0,               # 1: Block active (already normalized)
            5.0,               # 2: Block shape
            PAINT_MAX,         # 3: Block paint
            1.0,               # 4: Block illuminant (already normalized)
            1.0,               # 5: Block echo (already normalized)
            WALL_TYPE_MAX,     # 6: Wall type
            PAINT_MAX,         # 7: Wall paint
            1.0,               # 8: Wall illuminant (already normalized)
            1.0,               # 9: Wall echo (already normalized)
            LIQUID_TYPE_MAX,   # 10: Liquid type
            1.0,               # 11: Liquid amount (already normalized)
            1.0,               # 12: Red wire (already normalized)
            1.0,               # 13: Blue wire (already normalized)
            1.0,               # 14: Green wire (already normalized)
            1.0,               # 15: Yellow wire (already normalized)
            1.0,               # 16: Actuator (already normalized)
        ], dtype=torch.float32)
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tile tensor to [0, 1] range.
        
        Args:
            tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Normalized tensor of same shape
        """
        if tensor.dim() == 3:
            # Single sample (C, H, W)
            maxes = self.channel_maxes.view(-1, 1, 1).to(tensor.device)
        elif tensor.dim() == 4:
            # Batch (B, C, H, W)
            maxes = self.channel_maxes.view(1, -1, 1, 1).to(tensor.device)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
        
        return tensor / maxes
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tile tensor from [0, 1] range back to original scale.
        
        Args:
            tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Denormalized tensor of same shape
        """
        if tensor.dim() == 3:
            # Single sample (C, H, W)
            maxes = self.channel_maxes.view(-1, 1, 1).to(tensor.device)
        elif tensor.dim() == 4:
            # Batch (B, C, H, W)
            maxes = self.channel_maxes.view(1, -1, 1, 1).to(tensor.device)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
        
        return tensor * maxes
    
    def get_channel_weights(self, device='cpu') -> torch.Tensor:
        """
        Get per-channel weights for loss calculation.
        
        Channels with larger ranges get lower weights to balance the loss.
        
        Returns:
            Tensor of shape (17,) with per-channel weights
        """
        # Inverse of max values (normalized)
        weights = 1.0 / self.channel_maxes
        
        # Normalize weights so they sum to num_channels
        weights = weights * (len(weights) / weights.sum())
        
        return weights.to(device)


class WeightedMSELoss(torch.nn.Module):
    """
    MSE loss with per-channel weighting to handle different value ranges.
    """
    
    def __init__(self, normalizer: TileNormalizer):
        super().__init__()
        self.normalizer = normalizer
        self.register_buffer('weights', normalizer.get_channel_weights())
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            input: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            
        Returns:
            Scalar loss value
        """
        # Compute per-channel MSE
        mse_per_channel = torch.mean((input - target) ** 2, dim=(0, 2, 3))
        
        # Move weights to same device as input
        weights: torch.Tensor = self.weights.to(input.device)  # type: ignore[assignment]
        
        # Weight and sum
        weighted_loss = torch.sum(mse_per_channel * weights)
        
        return weighted_loss


# Global normalizer instance
_normalizer = None

def get_normalizer() -> TileNormalizer:
    """Get or create the global normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = TileNormalizer()
    return _normalizer
