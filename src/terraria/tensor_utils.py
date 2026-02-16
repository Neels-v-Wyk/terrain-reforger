"""
Utility functions for converting between tensor representations and human-readable formats.

Block and wall names are dynamically loaded from lihzahrd enums to ensure
complete coverage and automatic updates.
"""

import numpy as np
from typing import Dict, Any
import lihzahrd.enums as enums


def _format_enum_name(name: str) -> str:
    """Convert UPPER_SNAKE_CASE enum name to Title Case."""
    return name.replace('_', ' ').title()


# Generate complete dictionaries from lihzahrd enums
BLOCK_NAMES = {
    int(bt.value): _format_enum_name(bt.name) 
    for bt in enums.BlockType
}
BLOCK_NAMES[0] = "Air"  # Add air as block 0

WALL_NAMES = {
    int(wt.value): _format_enum_name(wt.name.replace('_UNSAFE', ''))
    for wt in enums.WallType
}
WALL_NAMES[0] = "No Wall"  # Add no wall as 0

LIQUID_TYPES = {
    int(lt.value): _format_enum_name(lt.name.replace('NO_LIQUID', 'None'))
    for lt in enums.LiquidType
}

BLOCK_SHAPES = {
    0: "Full",
    1: "Half",
    2: "Top-Right Slope",
    3: "Top-Left Slope",
    4: "Bottom-Right Slope",
    5: "Bottom-Left Slope"
}


def decode_tile_from_tensor(tensor_slice: np.ndarray) -> Dict[str, Any]:
    """
    Convert a single tile's tensor representation back to human-readable format.
    
    Args:
        tensor_slice: Array of shape (17,) representing a single tile
        
    Returns:
        Dictionary with human-readable tile information
    """
    block_type = int(tensor_slice[0])
    wall_type = int(tensor_slice[6])
    liquid_type = int(tensor_slice[10])
    
    result = {
        'block': {
            'type_id': block_type,
            'type_name': BLOCK_NAMES.get(block_type, f"Unknown Block {block_type}"),
            'active': bool(round(tensor_slice[1])),
            'shape': BLOCK_SHAPES.get(int(tensor_slice[2]), f"Unknown Shape {int(tensor_slice[2])}"),
            'paint': int(tensor_slice[3]) if tensor_slice[3] > 0 else None,
            'illuminant': bool(round(tensor_slice[4])),
            'echo': bool(round(tensor_slice[5]))
        },
        'wall': {
            'type_id': wall_type,
            'type_name': WALL_NAMES.get(wall_type, f"Unknown Wall {wall_type}"),
            'paint': int(tensor_slice[7]) if tensor_slice[7] > 0 else None,
            'illuminant': bool(round(tensor_slice[8])),
            'echo': bool(round(tensor_slice[9]))
        },
        'liquid': {
            'type': LIQUID_TYPES.get(liquid_type, f"Unknown Liquid {liquid_type}"),
            'amount': round(tensor_slice[11] * 255)
        },
        'wiring': {
            'red_wire': bool(round(tensor_slice[12])),
            'blue_wire': bool(round(tensor_slice[13])),
            'green_wire': bool(round(tensor_slice[14])),
            'yellow_wire': bool(round(tensor_slice[15])),
            'actuator': bool(round(tensor_slice[16]))
        }
    }
    
    return result


def format_tile_readable(tile_data: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format tile data into a compact readable string.
    
    Args:
        tile_data: Dictionary from decode_tile_from_tensor
        verbose: If True, include all properties. If False, only show active features.
        
    Returns:
        Formatted string representation
    """
    parts = []
    
    # Block info
    if tile_data['block']['active']:
        block_str = f"Block: {tile_data['block']['type_name']}"
        if tile_data['block']['shape'] != 'Full':
            block_str += f" ({tile_data['block']['shape']})"
        if tile_data['block']['paint']:
            block_str += f" [Paint:{tile_data['block']['paint']}]"
        parts.append(block_str)
    
    # Wall info
    if tile_data['wall']['type_id'] > 0:
        wall_str = f"Wall: {tile_data['wall']['type_name']}"
        if tile_data['wall']['paint']:
            wall_str += f" [Paint:{tile_data['wall']['paint']}]"
        parts.append(wall_str)
    
    # Liquid info
    if tile_data['liquid']['amount'] > 0:
        parts.append(f"Liquid: {tile_data['liquid']['type']} ({tile_data['liquid']['amount']}/255)")
    
    # Wiring (only if present)
    wires = []
    if tile_data['wiring']['red_wire']:
        wires.append('Red')
    if tile_data['wiring']['blue_wire']:
        wires.append('Blue')
    if tile_data['wiring']['green_wire']:
        wires.append('Green')
    if tile_data['wiring']['yellow_wire']:
        wires.append('Yellow')
    if wires:
        parts.append(f"Wires: {'+'.join(wires)}")
    if tile_data['wiring']['actuator']:
        parts.append("Actuator")
    
    return " | ".join(parts) if parts else "Empty tile"


def compare_tiles(original: np.ndarray, reconstructed: np.ndarray, position: tuple[int, int] | None = None) -> str:
    """
    Compare original and reconstructed tile tensors and return a formatted comparison.
    
    Args:
        original: Original tensor slice (17,)
        reconstructed: Reconstructed tensor slice (17,)
        position: Optional (x, y) position
        
    Returns:
        Formatted comparison string
    """
    orig_tile = decode_tile_from_tensor(original)
    recon_tile = decode_tile_from_tensor(reconstructed)
    
    pos_str = f"Tile at {position}: " if position else "Tile: "
    
    # Format both
    orig_str = format_tile_readable(orig_tile)
    recon_str = format_tile_readable(recon_tile)
    
    output = [pos_str]
    output.append(f"  Original:      {orig_str}")
    output.append(f"  Reconstructed: {recon_str}")
    
    # Highlight differences
    differences = []
    
    # Block differences
    if orig_tile['block']['active'] != recon_tile['block']['active']:
        differences.append(f"Block active: {orig_tile['block']['active']} → {recon_tile['block']['active']}")
    elif orig_tile['block']['active']:
        if orig_tile['block']['type_id'] != recon_tile['block']['type_id']:
            differences.append(
                f"Block type: {orig_tile['block']['type_name']} → {recon_tile['block']['type_name']}"
            )
        if orig_tile['block']['shape'] != recon_tile['block']['shape']:
            differences.append(f"Shape: {orig_tile['block']['shape']} → {recon_tile['block']['shape']}")
    
    # Wall differences
    if orig_tile['wall']['type_id'] != recon_tile['wall']['type_id']:
        differences.append(
            f"Wall: {orig_tile['wall']['type_name']} → {recon_tile['wall']['type_name']}"
        )
    
    # Liquid differences
    if orig_tile['liquid']['type'] != recon_tile['liquid']['type']:
        differences.append(f"Liquid: {orig_tile['liquid']['type']} → {recon_tile['liquid']['type']}")
    
    if differences:
        output.append(f"  Differences: {', '.join(differences)}")
    else:
        output.append("  ✓ Match")
    
    return "\n".join(output)


def summarize_chunk_comparison(original: np.ndarray, reconstructed: np.ndarray, 
                               sample_size: int = 10) -> str:
    """
    Summarize differences between original and reconstructed chunks.
    
    Args:
        original: Original tensor of shape (C, H, W) or (H, W, C)
        reconstructed: Reconstructed tensor of same shape
        sample_size: Number of example tiles to show
        
    Returns:
        Formatted summary string
    """
    # Handle both (C, H, W) and (H, W, C) formats
    if original.shape[0] == 17:
        # Convert from (C, H, W) to (H, W, C)
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
    
    h, w, c = original.shape
    
    output = []
    output.append("=" * 80)
    output.append(f"Chunk Comparison Summary (Shape: {h}x{w}, Channels: {c})")
    output.append("=" * 80)
    
    # Calculate statistics
    mse_per_channel = np.mean((original - reconstructed) ** 2, axis=(0, 1))
    
    channel_names = [
        "Block Type", "Block Active", "Block Shape", "Block Paint",
        "Block Illuminant", "Block Echo", "Wall Type", "Wall Paint",
        "Wall Illuminant", "Wall Echo", "Liquid Type", "Liquid Amount",
        "Red Wire", "Blue Wire", "Green Wire", "Yellow Wire", "Actuator"
    ]
    
    output.append("\nMean Squared Error per Channel:")
    for i, (name, mse) in enumerate(zip(channel_names, mse_per_channel)):
        output.append(f"  {i:2d}. {name:20s}: {mse:.6f}")
    
    output.append(f"\nOverall MSE: {np.mean(mse_per_channel):.6f}")
    
    # Sample some tiles with largest errors
    total_error = np.sum((original - reconstructed) ** 2, axis=2)
    flat_indices = np.argsort(total_error.flatten())[::-1][:sample_size]
    
    output.append(f"\nTop {sample_size} tiles with largest reconstruction errors:")
    output.append("-" * 80)
    
    for idx in flat_indices[:sample_size]:
        i = idx // w
        j = idx % w
        error = total_error[i, j]
        output.append(f"\nError: {error:.4f}")
        output.append(compare_tiles(original[i, j], reconstructed[i, j], position=(j, i)))
    
    return "\n".join(output)
