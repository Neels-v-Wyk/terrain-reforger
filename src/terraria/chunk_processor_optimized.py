"""
Optimized chunk processor for 8-channel format.
"""

import numpy as np
import lihzahrd
from .converters_optimized import tile_to_optimized_array


def extract_optimized_chunk(
    world: lihzahrd.World,
    x_start: int,
    y_start: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Extract a chunk from the world in optimized 8-channel format.
    
    Args:
        world: Loaded lihzahrd World object
        x_start: Starting X coordinate
        y_start: Starting Y coordinate  
        width: Chunk width in tiles
        height: Chunk height in tiles
        
    Returns:
        numpy array of shape (height, width, 8) with dtype float32
    """
    chunk = np.zeros((height, width, 8), dtype=np.float32)
    
    for y in range(height):
        world_y = y_start + y
        if world_y < 0 or world_y >= world.size.y:
            continue
            
        for x in range(width):
            world_x = x_start + x
            if world_x < 0 or world_x >= world.size.x:
                continue
            
            tile = world.tiles[world_x, world_y]
            chunk[y, x, :] = tile_to_optimized_array(tile)
    
    return chunk
