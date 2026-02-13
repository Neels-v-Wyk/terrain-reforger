"""
Utility functions for terrain data processing.

This module provides helper functions for common operations on Terraria
world data, such as visualization, statistics, and data validation.
"""

from typing import Dict, Any
import numpy as np

from .data_structures import ChunkData, TileData


# Shape names for human-readable output
SHAPE_NAMES = ['normal', 'half', 'slope_TR', 'slope_TL', 'slope_BR', 'slope_BL']

# Liquid type names
LIQUID_NAMES = ['none', 'water', 'lava', 'honey']


def format_tile_info(tile: TileData) -> str:
    """
    Format tile information as human-readable string.
    
    Args:
        tile: TileData object
        
    Returns:
        Formatted string describing the tile
    """
    lines = [f"Tile at ({tile.x}, {tile.y}):"]
    
    if tile.block_active:
        shape_name = (
            SHAPE_NAMES[tile.block_shape] 
            if tile.block_shape < len(SHAPE_NAMES) 
            else str(tile.block_shape)
        )
        lines.append(f"  Block: Type {tile.block_type}, Shape: {shape_name}")
        
        if tile.block_paint:
            lines.append(f"    Paint: Color {tile.block_paint}")
        if tile.block_illuminant:
            lines.append(f"    Illuminant: Yes")
        if tile.block_echo:
            lines.append(f"    Echo: Yes")
    
    if tile.wall_type:
        lines.append(f"  Wall: Type {tile.wall_type}")
        if tile.wall_paint:
            lines.append(f"    Paint: Color {tile.wall_paint}")
            
    if tile.liquid_type > 0:
        liquid_name = (
            LIQUID_NAMES[tile.liquid_type]
            if tile.liquid_type < len(LIQUID_NAMES)
            else str(tile.liquid_type)
        )
        lines.append(f"  Liquid: {liquid_name} (amount: {tile.liquid_amount})")
    
    # Check for any wiring
    wires = []
    if tile.wire_red:
        wires.append('red')
    if tile.wire_blue:
        wires.append('blue')
    if tile.wire_green:
        wires.append('green')
    if tile.wire_yellow:
        wires.append('yellow')
    if tile.actuator:
        wires.append('actuator')
    
    if wires:
        lines.append(f"  Wiring: {', '.join(wires)}")
    
    return '\n'.join(lines)


def get_tensor_statistics(tensor: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a tensor.
    
    Args:
        tensor: Tensor from ChunkData.to_tensor() with shape (H, W, 17)
        
    Returns:
        Dictionary with statistics about the tensor content
    """
    h, w, c = tensor.shape
    total_tiles = h * w
    
    return {
        'shape': (h, w, c),
        'dtype': str(tensor.dtype),
        'memory_kb': tensor.nbytes / 1024,
        'tiles': {
            'total': total_tiles,
            'with_blocks': int(np.count_nonzero(tensor[:, :, 0])),
            'with_walls': int(np.count_nonzero(tensor[:, :, 6])),
            'with_liquid': int(np.count_nonzero(tensor[:, :, 10])),
            'with_slopes': int(np.count_nonzero(tensor[:, :, 2])),
            'with_paint': int(np.count_nonzero(tensor[:, :, 3]) + np.count_nonzero(tensor[:, :, 7])),
        },
        'block_types': {
            'unique': int(len(np.unique(tensor[:, :, 0])) - 1),  # -1 for zero
            'most_common': int(np.bincount(tensor[:, :, 0].astype(int).flatten()).argmax()),
        },
        'liquids': {
            'water_tiles': int(np.sum(tensor[:, :, 10] == 1)),
            'lava_tiles': int(np.sum(tensor[:, :, 10] == 2)),
            'honey_tiles': int(np.sum(tensor[:, :, 10] == 3)),
            'avg_amount': float(np.mean(tensor[:, :, 11][tensor[:, :, 11] > 0])) if np.any(tensor[:, :, 11] > 0) else 0.0,
        },
    }


def validate_chunk(chunk: ChunkData) -> Dict[str, Any]:
    """
    Validate chunk data for common issues.
    
    Args:
        chunk: ChunkData to validate
        
    Returns:
        Dictionary with validation results and warnings
    """
    issues = []
    warnings = []
    
    # Check dimensions
    if chunk.width <= 0 or chunk.height <= 0:
        issues.append("Invalid chunk dimensions")
    
    # Check tile count
    expected_tiles = chunk.width * chunk.height
    actual_tiles = sum(len(row) for row in chunk.tiles)
    
    if len(chunk.tiles) != chunk.height:
        issues.append(f"Row count mismatch: expected {chunk.height}, got {len(chunk.tiles)}")
    
    for i, row in enumerate(chunk.tiles):
        if len(row) != chunk.width:
            issues.append(f"Row {i} width mismatch: expected {chunk.width}, got {len(row)}")
    
    # Check for suspicious values
    for row in chunk.tiles:
        for tile in row:
            if tile.block_shape < 0 or tile.block_shape > 5:
                warnings.append(f"Tile ({tile.x}, {tile.y}): Invalid block shape {tile.block_shape}")
            
            if tile.liquid_type < 0 or tile.liquid_type > 3:
                warnings.append(f"Tile ({tile.x}, {tile.y}): Invalid liquid type {tile.liquid_type}")
            
            if tile.liquid_amount < 0 or tile.liquid_amount > 255:
                warnings.append(f"Tile ({tile.x}, {tile.y}): Invalid liquid amount {tile.liquid_amount}")
            
            if tile.block_paint and (tile.block_paint < 0 or tile.block_paint > 31):
                warnings.append(f"Tile ({tile.x}, {tile.y}): Invalid block paint {tile.block_paint}")
            
            if tile.wall_paint and (tile.wall_paint < 0 or tile.wall_paint > 31):
                warnings.append(f"Tile ({tile.x}, {tile.y}): Invalid wall paint {tile.wall_paint}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings[:10],  # Limit warnings to first 10
        'warning_count': len(warnings),
    }


def print_chunk_summary(chunk: ChunkData, show_entities: bool = True):
    """
    Print a formatted summary of chunk contents.
    
    Args:
        chunk: ChunkData to summarize
        show_entities: Whether to show chest/sign/entity counts
    """
    summary = chunk.get_summary()
    
    print("=" * 60)
    print(f"Chunk at ({chunk.x_start}, {chunk.y_start})")
    print(f"Size: {chunk.width} x {chunk.height} = {summary['total_tiles']} tiles")
    print("=" * 60)
    
    print(f"\nTerrain:")
    print(f"  Active blocks: {summary['active_blocks']:,}")
    print(f"  Slopes: {summary['slopes']:,}")
    print(f"  Walls: {summary['walls']:,}")
    print(f"  Liquids: {summary['liquids']:,}")
    
    if show_entities:
        print(f"\nEntities:")
        print(f"  Chests: {summary['chests']}")
        print(f"  Signs: {summary['signs']}")
        print(f"  Tile Entities: {summary['tile_entities']}")
    
    print()
