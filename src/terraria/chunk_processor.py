"""
Chunk extraction and processing for Terraria worlds.

This module provides functions to extract chunks from Terraria world files
for machine learning training and generation.
"""

from typing import Optional, Tuple
import lihzahrd

from .data_structures import ChunkData, TileData
from .converters import tile_to_data


# Default chunk parameters
DEFAULT_CHUNK_SIZE = 64
DEFAULT_OVERLAP = 0


def extract_chunk(
    world: lihzahrd.World,
    x_start: int,
    y_start: int,
    width: int = DEFAULT_CHUNK_SIZE,
    height: int = DEFAULT_CHUNK_SIZE
) -> ChunkData:
    """
    Extract a chunk from a Terraria world.
    
    Args:
        world: Loaded Terraria world object
        x_start: Starting x-coordinate (top-left corner)
        y_start: Starting y-coordinate (top-left corner)
        width: Width of the chunk in tiles
        height: Height of the chunk in tiles
        
    Returns:
        ChunkData object containing all tile information and entities
        
    Raises:
        IndexError: If coordinates are outside world bounds
    """
    # Validate coordinates
    if x_start < 0 or y_start < 0:
        raise ValueError("Chunk coordinates must be non-negative")
    if x_start + width > world.size.x or y_start + height > world.size.y:
        raise ValueError(
            f"Chunk extends beyond world bounds. "
            f"World size: {world.size}, Chunk: ({x_start}, {y_start}, {width}, {height})"
        )
    
    chunk_data = ChunkData(
        x_start=x_start,
        y_start=y_start,
        width=width,
        height=height
    )
    
    # Extract tiles
    tiles = []
    for i in range(y_start, y_start + height):
        row = []
        for j in range(x_start, x_start + width):
            tile = world.tiles[j, i]
            tile_data = tile_to_data(tile, j, i)
            row.append(tile_data)
        tiles.append(row)
    
    chunk_data.tiles = tiles
    
    # Extract entities within chunk bounds
    x_end = x_start + width
    y_end = y_start + height
    
    # Include chests
    for chest in world.chests:
        if x_start <= chest.position.x < x_end and y_start <= chest.position.y < y_end:
            chunk_data.chests.append(chest)
    
    # Include signs
    for sign in world.signs:
        if x_start <= sign.position.x < x_end and y_start <= sign.position.y < y_end:
            chunk_data.signs.append(sign)
    
    # Include tile entities
    for tile_entity in world.tile_entities:
        if x_start <= tile_entity.position.x < x_end and y_start <= tile_entity.position.y < y_end:
            chunk_data.tile_entities.append(tile_entity)
    
    return chunk_data


def iter_chunks(
    world: lihzahrd.World,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    region: Optional[Tuple[int, int, int, int]] = None
):
    """
    Iterate over chunks in a sliding window fashion across the world.
    
    Args:
        world: Loaded Terraria world object
        chunk_size: Size of each chunk (assumes square chunks)
        overlap: Number of tiles to overlap between adjacent chunks
        region: Optional (x_start, y_start, x_end, y_end) to limit iteration
        
    Yields:
        ChunkData objects for each position in the grid
        
    Example:
        >>> world = load_world("world.wld")
        >>> for chunk in iter_chunks(world, chunk_size=64, overlap=8):
        ...     tensor = chunk.to_tensor()
        ...     # Process tensor...
    """
    if region:
        x_start, y_start, x_end, y_end = region
    else:
        x_start, y_start = 0, 0
        x_end, y_end = world.size.x, world.size.y
    
    stride = chunk_size - overlap
    
    y = y_start
    while y + chunk_size <= y_end:
        x = x_start
        while x + chunk_size <= x_end:
            yield extract_chunk(world, x, y, chunk_size, chunk_size)
            x += stride
        y += stride
