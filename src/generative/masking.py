"""
Masking utilities for MaskGIT training.

Implements various masking strategies for masked token prediction training.
"""

import torch
import math
from typing import Tuple, Optional


def random_masking(
    tokens: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply random masking to token sequences.
    
    Args:
        tokens: (batch, seq_len) ground truth token sequences
        mask_ratio: Fraction of tokens to mask (0.0-1.0)
        mask_token_id: ID of the MASK token
        
    Returns:
        masked_tokens: (batch, seq_len) tokens with some positions masked
        mask: (batch, seq_len) boolean mask indicating which positions are masked
        targets: (batch, seq_len) original tokens (for computing loss)
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Determine number of tokens to mask
    num_masked = int(seq_len * mask_ratio)
    num_masked = max(1, num_masked)  # Mask at least 1 token
    
    # Create random permutation for each sample
    noise = torch.rand(batch_size, seq_len, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    # Create mask: True for masked positions
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_masked], True)
    
    # Create masked token sequence
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id
    
    return masked_tokens, mask, tokens


def scheduled_masking(
    tokens: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
    epoch: int,
    total_epochs: int,
    start_ratio: float = 0.6,
    end_ratio: float = 0.3,
    schedule: str = 'cosine',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply masking with scheduled mask ratio that decreases over training.
    
    Curriculum learning: start with more masking (harder), gradually reduce (easier).
    This helps the model learn better representations.
    
    Args:
        tokens: (batch, seq_len) ground truth token sequences
        mask_ratio: Base mask ratio (will be modulated by schedule)
        mask_token_id: ID of the MASK token
        epoch: Current training epoch
        total_epochs: Total number of training epochs
        start_ratio: Starting mask ratio (default 0.6 = 60%)
        end_ratio: Ending mask ratio (default 0.3 = 30%)
        schedule: Schedule type ('linear', 'cosine', 'constant')
        
    Returns:
        masked_tokens: (batch, seq_len) tokens with some positions masked
        mask: (batch, seq_len) boolean mask
        targets: (batch, seq_len) original tokens
    """
    # Compute scheduled mask ratio
    progress = epoch / max(total_epochs, 1)
    
    if schedule == 'linear':
        current_ratio = start_ratio + (end_ratio - start_ratio) * progress
    elif schedule == 'cosine':
        # Smooth cosine decay
        current_ratio = end_ratio + (start_ratio - end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == 'constant':
        current_ratio = mask_ratio
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    # Apply random masking with computed ratio
    return random_masking(tokens, current_ratio, mask_token_id)


def block_masking(
    tokens: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
    grid_size: int = 8,
    block_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply block-based masking for 2D spatial structures.
    
    Instead of masking individual tokens, masks contiguous 2D blocks.
    This is more challenging and encourages learning spatial coherence.
    
    Args:
        tokens: (batch, seq_len) ground truth tokens (flattened 2D grid)
        mask_ratio: Target fraction of tokens to mask
        mask_token_id: ID of the MASK token
        grid_size: Size of 2D grid (8 for 8×8)
        block_size: Size of square blocks to mask (2 = 2×2 blocks)
        
    Returns:
        masked_tokens: (batch, seq_len) tokens with blocks masked
        mask: (batch, seq_len) boolean mask
        targets: (batch, seq_len) original tokens
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    assert seq_len == grid_size ** 2, f"seq_len {seq_len} != grid_size^2 {grid_size**2}"
    
    # Reshape to 2D grid
    tokens_2d = tokens.view(batch_size, grid_size, grid_size)
    
    # Calculate how many blocks to mask
    num_blocks_per_row = grid_size // block_size
    total_blocks = num_blocks_per_row ** 2
    num_blocks_to_mask = int(total_blocks * mask_ratio)
    num_blocks_to_mask = max(1, num_blocks_to_mask)
    
    # Create mask in 2D
    mask_2d = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        # Randomly select blocks to mask
        block_indices = torch.randperm(total_blocks, device=device)[:num_blocks_to_mask]
        
        for block_idx in block_indices:
            # Convert 1D block index to 2D block coordinates
            block_row = (block_idx // num_blocks_per_row).item()
            block_col = (block_idx % num_blocks_per_row).item()
            
            # Mark all tokens in this block as masked
            row_start = block_row * block_size
            row_end = row_start + block_size
            col_start = block_col * block_size
            col_end = col_start + block_size
            
            mask_2d[b, row_start:row_end, col_start:col_end] = True
    
    # Flatten back to 1D
    mask = mask_2d.view(batch_size, seq_len)
    
    # Create masked tokens
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id
    
    return masked_tokens, mask, tokens


def get_mask_ratio_for_epoch(
    epoch: int,
    total_epochs: int,
    start_ratio: float = 0.6,
    end_ratio: float = 0.3,
    schedule: str = 'cosine',
) -> float:
    """
    Get the mask ratio for a given epoch based on schedule.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        start_ratio: Starting mask ratio
        end_ratio: Ending mask ratio
        schedule: Schedule type
        
    Returns:
        mask_ratio: Mask ratio for this epoch
    """
    progress = epoch / max(total_epochs, 1)
    
    if schedule == 'linear':
        return start_ratio + (end_ratio - start_ratio) * progress
    elif schedule == 'cosine':
        return end_ratio + (start_ratio - end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == 'constant':
        return start_ratio
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
