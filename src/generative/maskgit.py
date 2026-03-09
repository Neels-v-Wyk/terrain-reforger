"""
MaskGIT: Masked Generative Image Transformer for terrain token generation.

MaskGIT uses bidirectional masked prediction instead of autoregressive generation,
enabling parallel token generation that is 5-6× faster while maintaining quality.

Key differences from autoregressive:
- Bidirectional self-attention (BERT-style) instead of causal masking
- Masked token prediction during training (like BERT MLM)
- Iterative parallel decoding during generation (predict all positions simultaneously)
"""

import math
from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for 8×8 grids.
    
    Uses factorized embeddings: pos_emb(i,j) = row_emb(i) + col_emb(j)
    This preserves the 2D spatial structure better than 1D positions.
    """
    
    def __init__(self, d_model: int, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        
        # Split embedding dimension between row and column
        d_row = d_model // 2
        d_col = d_model - d_row
        
        self.row_embed = nn.Embedding(grid_size, d_row)
        self.col_embed = nn.Embedding(grid_size, d_col)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D positional encodings for a flattened 8×8 grid.
        
        Args:
            seq_len: Sequence length (should be 64 for 8×8 grid)
            device: Device to create tensor on
            
        Returns:
            pos_encoding: (seq_len, d_model) positional encodings
        """
        assert seq_len == self.grid_size ** 2, \
            f"Expected seq_len={self.grid_size**2}, got {seq_len}"
        
        # Create 2D grid positions in raster scan order
        row_idx = torch.arange(self.grid_size, device=device).repeat_interleave(self.grid_size)
        col_idx = torch.arange(self.grid_size, device=device).repeat(self.grid_size)
        
        # Get embeddings
        row_emb = self.row_embed(row_idx)  # (64, d_row)
        col_emb = self.col_embed(col_idx)  # (64, d_col)
        
        # Concatenate
        pos_encoding = torch.cat([row_emb, col_emb], dim=-1)  # (64, d_model)
        
        return pos_encoding


class TerrainMaskGIT(nn.Module):
    """
    MaskGIT transformer for terrain token generation.
    
    Architecture:
    - Bidirectional encoder (no causal masking)
    - Masked token prediction during training
    - Iterative parallel decoding during generation
    
    Args:
        vocab_size: Size of token vocabulary (codebook size, typically 1024)
        seq_len: Sequence length (64 for 8×8 grid)
        d_model: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout: Dropout probability
        grid_size: Size of 2D grid (8 for 8×8)
        mask_token_id: Special token ID for MASK (defaults to vocab_size)
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        seq_len: int = 64,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        grid_size: int = 8,
        mask_token_id: Optional[int] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.grid_size = grid_size
        
        # MASK token is a special token beyond the vocabulary
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size
        
        # Token embeddings (vocab_size + 1 to include MASK token)
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        
        # 2D positional encoding (reuse from transformer.py)
        self.pos_encoding = PositionalEncoding2D(d_model, grid_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional transformer encoder layers (no causal masking)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection to vocabulary (not including MASK token in output)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        # Token embeddings
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        
        # Output projection
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self, 
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            tokens: (batch, seq_len) token indices (may contain MASK tokens)
            mask: Optional (batch, seq_len) boolean mask indicating which positions
                  to compute loss on (True = compute loss, False = ignore)
            
        Returns:
            logits: (batch, seq_len, vocab_size) predicted logits for all positions
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Token embeddings
        tok_emb = self.token_embed(tokens)  # (batch, seq_len, d_model)
        
        # Add 2D positional encoding
        pos_emb = self.pos_encoding(seq_len, device)  # (seq_len, d_model)
        x = tok_emb + pos_emb.unsqueeze(0)  # (batch, seq_len, d_model)
        
        # Dropout
        x = self.dropout(x)
        
        # Bidirectional transformer (no causal mask)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def compute_loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked token prediction loss.
        
        Args:
            tokens: (batch, seq_len) input tokens (with MASK tokens)
            targets: (batch, seq_len) ground truth tokens
            mask: (batch, seq_len) boolean mask (True = compute loss on this position)
            
        Returns:
            loss: Scalar cross-entropy loss on masked positions only
        """
        logits = self.forward(tokens)  # (batch, seq_len, vocab_size)
        
        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits[mask],  # (num_masked, vocab_size)
            targets[mask],  # (num_masked,)
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        num_iterations: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        schedule: str = 'cosine',
        device: Optional[torch.device] = None,
        progress_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Generate token sequences using iterative parallel decoding.
        
        Algorithm:
        1. Start with all tokens masked
        2. For each iteration:
           - Predict logits for all positions
           - Compute confidence (max probability) for each masked position
           - Unmask the most confident tokens according to schedule
           - Sample from the distribution for unmasked positions
        3. Return final tokens after all iterations
        
        Args:
            num_samples: Number of sequences to generate
            num_iterations: Number of decoding iterations (8-16 recommended)
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling
            schedule: Unmasking schedule ('linear', 'cosine', 'exponential')
            device: Device to generate on
            progress_callback: Optional callback(iteration, tokens) for visualization
            
        Returns:
            tokens: (num_samples, seq_len) generated token sequences
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Start with all positions masked
        tokens = torch.full(
            (num_samples, self.seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which positions are still masked
        is_masked = torch.ones(
            (num_samples, self.seq_len),
            dtype=torch.bool,
            device=device
        )
        
        # Iterative decoding
        for iter_idx in range(num_iterations):
            # Forward pass to get predictions for all positions
            logits = self.forward(tokens)  # (num_samples, seq_len, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, :, 0] = False  # Keep at least one token
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    2, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Get probabilities and confidence
            probs = F.softmax(logits, dim=-1)  # (num_samples, seq_len, vocab_size)
            confidence, predicted_tokens = probs.max(dim=-1)  # (num_samples, seq_len)
            
            # Determine how many tokens to unmask this iteration
            num_masked = is_masked.sum(dim=-1)  # (num_samples,)
            num_to_unmask = self._get_num_to_unmask(
                iteration=iter_idx,
                total_iterations=num_iterations,
                num_masked=num_masked,
                schedule=schedule
            )
            
            # For each sample, unmask the most confident tokens
            for sample_idx in range(num_samples):
                if num_masked[sample_idx] == 0:
                    continue  # All tokens already unmasked
                
                # Get confidence of masked positions for this sample
                masked_positions = is_masked[sample_idx]
                masked_confidence = confidence[sample_idx].clone()
                masked_confidence[~masked_positions] = -float('inf')  # Ignore unmasked
                
                # Get top-k most confident positions to unmask
                k = min(num_to_unmask[sample_idx].item(), masked_positions.sum().item())
                if k > 0:
                    _, top_indices = torch.topk(masked_confidence, k)
                    
                    # Sample tokens for these positions (use predicted token or sample from distribution)
                    for idx in top_indices:
                        # Sample from the distribution
                        sampled_token = torch.multinomial(probs[sample_idx, idx], num_samples=1)
                        tokens[sample_idx, idx] = sampled_token
                        is_masked[sample_idx, idx] = False
            
            # Callback for progress visualization
            if progress_callback is not None:
                progress_callback(iter_idx + 1, tokens.clone())
        
        return tokens
    
    def _get_num_to_unmask(
        self,
        iteration: int,
        total_iterations: int,
        num_masked: torch.Tensor,
        schedule: str = 'cosine',
    ) -> torch.Tensor:
        """
        Determine how many tokens to unmask at this iteration.
        
        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total number of iterations
            num_masked: (batch,) number of masked tokens remaining per sample
            schedule: Unmasking schedule
            
        Returns:
            num_to_unmask: (batch,) number of tokens to unmask per sample
        """
        ratio = (iteration + 1) / total_iterations
        
        if schedule == 'linear':
            # Linear: unmask constant fraction each iteration
            unmask_ratio = 1.0 / total_iterations
        elif schedule == 'cosine':
            # Cosine: slow start, fast middle, slow end
            # This schedule is proven to work well in MaskGIT paper
            unmask_ratio = math.cos((1 - ratio) * math.pi / 2)
            prev_unmask_ratio = math.cos((1 - (iteration / total_iterations)) * math.pi / 2)
            unmask_ratio = unmask_ratio - prev_unmask_ratio
        elif schedule == 'exponential':
            # Exponential: fast start, slow end
            unmask_ratio = 1 - (1 - ratio) ** 2
            prev_unmask_ratio = 1 - (1 - (iteration / total_iterations)) ** 2
            unmask_ratio = unmask_ratio - prev_unmask_ratio
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # For last iteration, unmask everything remaining
        if iteration == total_iterations - 1:
            return num_masked
        
        # Calculate number to unmask (at least 1 per iteration)
        num_to_unmask = torch.ceil(num_masked.float() * unmask_ratio).long()
        num_to_unmask = torch.clamp(num_to_unmask, min=1)
        
        return num_to_unmask
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, don't count embedding parameters
            
        Returns:
            num_params: Total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embed.weight.numel()
        return n_params


def create_maskgit_model(
    vocab_size: int = 1024,
    seq_len: int = 64,
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> TerrainMaskGIT:
    """
    Factory function to create a TerrainMaskGIT model.
    
    Default configuration:
    - 12 layers, 512 dim, 8 heads (slightly deeper than autoregressive)
    - ~65M parameters
    - Optimized for quality/speed balance
    """
    model = TerrainMaskGIT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    )
    
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print(f"Created TerrainMaskGIT:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-embedding parameters: {non_emb_params:,}")
    print(f"  Architecture: {n_layers} layers, {d_model} dim, {n_heads} heads")
    print(f"  MASK token ID: {model.mask_token_id}")
    
    return model


# Preset configurations (based on expert recommendations)
MASKGIT_SMALL_CONFIG = {
    'd_model': 256,
    'n_layers': 6,
    'n_heads': 4,
    'd_ff': 1024,
    'dropout': 0.1,
}

MASKGIT_MEDIUM_CONFIG = {
    'd_model': 512,
    'n_layers': 12,  # Deeper than autoregressive for better quality
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
}

MASKGIT_SMALL_PLUS_CONFIG = {
    'd_model': 384,
    'n_layers': 8,
    'n_heads': 6,
    'd_ff': 1536,
    'dropout': 0.1,
}

MASKGIT_LARGE_CONFIG = {
    'd_model': 768,
    'n_layers': 12,
    'n_heads': 12,
    'd_ff': 3072,
    'dropout': 0.1,
}
