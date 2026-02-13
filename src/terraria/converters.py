"""
Conversion utilities for Terraria world data.

This module provides functions to convert between lihzahrd library objects
and our internal data structures.
"""

from typing import Optional
import lihzahrd

from .data_structures import TileData


def tile_to_data(tile: lihzahrd.world.Tile, x: int, y: int) -> TileData:
    """
    Convert a lihzahrd Tile object to our structured TileData.
    
    Args:
        tile: Tile object from lihzahrd library
        x: X-coordinate of the tile
        y: Y-coordinate of the tile
        
    Returns:
        TileData object with all tile properties extracted
    """
    tile_data = TileData(x=x, y=y)
    
    # Extract block information
    if tile.block:
        tile_data.block_type = tile.block.type
        tile_data.block_active = tile.block.is_active
        tile_data.block_shape = (
            tile.block.shape.value 
            if hasattr(tile.block.shape, 'value') 
            else 0
        )
        tile_data.block_paint = tile.block.paint
        tile_data.block_illuminant = getattr(tile.block, 'is_illuminant', False)
        tile_data.block_echo = getattr(tile.block, 'is_echo', False)
    
    # Extract wall information
    if tile.wall:
        tile_data.wall_type = tile.wall.type
        tile_data.wall_paint = tile.wall.paint
        tile_data.wall_illuminant = getattr(tile.wall, 'is_illuminant', False)
        tile_data.wall_echo = getattr(tile.wall, 'is_echo', False)
    
    # Extract liquid information
    if tile.liquid:
        # Liquid type is an enum value (1=water, 2=lava, 3=honey)
        tile_data.liquid_type = (
            tile.liquid.type.value 
            if hasattr(tile.liquid.type, 'value') 
            else tile.liquid.type
        )
        tile_data.liquid_amount = getattr(tile.liquid, 'volume', 0)
    
    # Extract wiring information
    if tile.wiring:
        tile_data.wire_red = getattr(tile.wiring, 'red', False)
        tile_data.wire_blue = getattr(tile.wiring, 'blue', False)
        tile_data.wire_green = getattr(tile.wiring, 'green', False)
        tile_data.wire_yellow = getattr(tile.wiring, 'yellow', False)
        tile_data.actuator = getattr(tile.wiring, 'actuator', False)
    
    return tile_data
