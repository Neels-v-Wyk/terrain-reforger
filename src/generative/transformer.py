"""
MaskGIT: Masked Generative Image Transformer for terrain token generation.

This module now uses MaskGIT instead of autoregressive generation.
MaskGIT uses bidirectional masked prediction for 5-6× faster parallel generation.

For backward compatibility, this file re-exports MaskGIT classes.
"""

# Re-export MaskGIT as the primary transformer implementation
from .maskgit import (
    TerrainMaskGIT as TerrainTransformer,
    create_maskgit_model as create_model,
    PositionalEncoding2D,
    MASKGIT_SMALL_CONFIG as SMALL_CONFIG,
    MASKGIT_MEDIUM_CONFIG as MEDIUM_CONFIG,
    MASKGIT_SMALL_PLUS_CONFIG as SMALL_PLUS_CONFIG,
    MASKGIT_LARGE_CONFIG as LARGE_CONFIG,
)

__all__ = [
    'TerrainTransformer',
    'create_model',
    'PositionalEncoding2D',
    'SMALL_CONFIG',
    'MEDIUM_CONFIG',
    'SMALL_PLUS_CONFIG',
    'LARGE_CONFIG',
]
