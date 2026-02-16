"""Terraria world processing package (optimized pipeline)."""

from .world_handler import load_world
from .chunk_processor_optimized import extract_optimized_chunk
from .converters_optimized import tile_to_optimized_array, optimized_array_to_tile_dict

__all__ = [
    "load_world",
    "extract_optimized_chunk",
    "tile_to_optimized_array",
    "optimized_array_to_tile_dict",
]

__version__ = '0.1.0'
