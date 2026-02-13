"""
Terraria world processing for ML applications.

This package provides tools for loading, processing, and converting Terraria
world data into formats suitable for machine learning, specifically focused
on terrain generation using discrete diffusion models.

Main modules:
    - world_handler: World loading and management
    - data_structures: Core data structures (TileData, ChunkData)
    - converters: Conversion utilities between lihzahrd and internal formats
    - chunk_processor: Chunk extraction and iteration
    - utils: Helper functions for visualization and validation
"""

from .world_handler import load_world
from .data_structures import TileData, ChunkData
from .chunk_processor import extract_chunk, iter_chunks, DEFAULT_CHUNK_SIZE
from .utils import (
    format_tile_info,
    get_tensor_statistics,
    validate_chunk,
    print_chunk_summary,
)

__all__ = [
    # World loading
    'load_world',
    
    # Data structures
    'TileData',
    'ChunkData',
    
    # Chunk processing
    'extract_chunk',
    'iter_chunks',
    'DEFAULT_CHUNK_SIZE',
    
    # Utilities
    'format_tile_info',
    'get_tensor_statistics',
    'validate_chunk',
    'print_chunk_summary',
]

__version__ = '0.1.0'
