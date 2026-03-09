"""
Autoregressive Transformer for generating VQVAE token sequences.

This is a GPT-style decoder-only transformer designed to generate
8×8 grids of discrete tokens (flattened to 64-token sequences) that
will be decoded by the VQVAE to produce 32×32 terrain chunks.
"""

import math
from typing import Tuple, Optional

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


class TerrainTransformer(nn.Module):
    """
    GPT-style autoregressive transformer for terrain token generation.
    
    Architecture:
    - Decoder-only transformer (causal self-attention)
    - Generates 64 tokens sequentially in raster scan order
    - Uses 2D positional encoding to preserve spatial structure
    
    Args:
        vocab_size: Size of token vocabulary (codebook size)
        seq_len: Sequence length (64 for 8×8 grid)
        d_model: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout: Dropout probability
        grid_size: Size of 2D grid (8 for 8×8)
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        seq_len: int = 64,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        grid_size: int = 8,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.grid_size = grid_size
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, grid_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Create causal mask (upper triangular)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        )
        
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
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            tokens: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens for loss computation (optional)
            
        Returns:
            logits: (batch, seq_len, vocab_size) predicted logits
            loss: Cross-entropy loss if targets provided, else None
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
        
        # Transformer with causal masking
        # Note: PyTorch expects mask[i,j] = True to IGNORE position j when computing i
        x = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=self.causal_mask[:seq_len, :seq_len],
            tgt_is_causal=True,
        )
        
        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction='mean'
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate token sequences autoregressively.
        
        Args:
            num_samples: Number of sequences to generate
            temperature: Sampling temperature (lower = more conservative)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling (sample from top cumulative probability p)
            device: Device to generate on
            
        Returns:
            tokens: (num_samples, seq_len) generated token sequences
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Start with zeros (or could use a special start token)
        tokens = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=device)
        
        # Generate tokens one at a time
        for i in range(self.seq_len):
            # Forward pass
            logits, _ = self.forward(tokens)  # (num_samples, seq_len, vocab_size)
            
            # Get logits for current position
            next_logits = logits[:, i, :] / temperature  # (num_samples, vocab_size)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (num_samples, 1)
            
            # Update tokens
            if i < self.seq_len - 1:
                tokens[:, i + 1] = next_token.squeeze(-1)
        
        return tokens
    
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


def create_model(
    vocab_size: int = 1024,
    seq_len: int = 64,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> TerrainTransformer:
    """
    Factory function to create a TerrainTransformer model.
    
    Default configuration:
    - 8 layers, 512 dim, 8 heads
    - ~50M parameters
    - Good balance of quality and speed
    """
    model = TerrainTransformer(
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
    
    print(f"Created TerrainTransformer:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-embedding parameters: {non_emb_params:,}")
    print(f"  Architecture: {n_layers} layers, {d_model} dim, {n_heads} heads")
    
    return model


# Preset configurations
SMALL_CONFIG = {
    'd_model': 256,
    'n_layers': 6,
    'n_heads': 4,
    'd_ff': 1024,
    'dropout': 0.1,
}

MEDIUM_CONFIG = {
    'd_model': 512,
    'n_layers': 8,
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
}

LARGE_CONFIG = {
    'd_model': 768,
    'n_layers': 12,
    'n_heads': 12,
    'd_ff': 3072,
    'dropout': 0.1,
}
