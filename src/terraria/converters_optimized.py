"""
Optimized tile conversion for 8-channel natural world format.

Converts lihzahrd tiles to compact 8-channel representation:
0. block_type (index 0-217)
1. block_shape (index 0-5)
2. wall_type (index 0-76)
3. liquid_type (0-4: none, water, lava, honey, shimmer)
4. wire_red (0 or 1)
5. wire_blue (0 or 1)
6. wire_green (0 or 1)
7. actuator (0 or 1)
"""

import numpy as np
import lihzahrd
from .natural_ids import BLOCK_ID_TO_INDEX, WALL_ID_TO_INDEX

# Liquid type mapping
LIQUID_TYPE_MAP = {
    'WATER': 1,
    'LAVA': 2,
    'HONEY': 3,
    'SHIMMER': 4
}

def tile_to_optimized_array(tile: lihzahrd.world.Tile) -> np.ndarray:
    """
    Convert a lihzahrd tile to 8-element numpy array.
    
    Args:
        tile: Tile from lihzahrd library
        
    Returns:
        8-element float32 array
    """
    data = np.zeros(8, dtype=np.float32)
    
    # Block information
    if tile.block is not None and tile.block.is_active:
        # Get game block ID
        block_game_id = tile.block.type.value if hasattr(tile.block.type, 'value') else tile.block.type
        
        # Convert to compact index (0-217)
        if block_game_id in BLOCK_ID_TO_INDEX:
            data[0] = BLOCK_ID_TO_INDEX[block_game_id]
        else:
            # Unknown block, map to 0 (air/empty)
            data[0] = 0
        
        # Block shape (0-5, categorical)
        shape_val = tile.block.shape.value if hasattr(tile.block.shape, 'value') else tile.block.shape
        data[1] = float(shape_val) if shape_val <= 5 else 0.0
    
    # Wall information
    if tile.wall is not None:
        # Get game wall ID
        wall_game_id = tile.wall.type.value if hasattr(tile.wall.type, 'value') else tile.wall.type
        
        # Convert to compact index (0-76)
        if wall_game_id in WALL_ID_TO_INDEX:
            data[2] = WALL_ID_TO_INDEX[wall_game_id]
        else:
            # Unknown wall, map to 0
            data[2] = 0
    
    # Liquid information
    if tile.liquid is not None:
        liquid_volume = getattr(tile.liquid, 'volume', getattr(tile.liquid, 'amount', 0))
        if liquid_volume > 0:
            # Get liquid type
            liquid_type_name = tile.liquid.type.name if hasattr(tile.liquid.type, 'name') else str(tile.liquid.type)
            liquid_type_name = liquid_type_name.upper()
            data[3] = LIQUID_TYPE_MAP.get(liquid_type_name, 0)
    
    # Wiring information
    if tile.wiring is not None:
        data[4] = 1.0 if getattr(tile.wiring, 'red', False) else 0.0
        data[5] = 1.0 if getattr(tile.wiring, 'blue', False) else 0.0
        data[6] = 1.0 if getattr(tile.wiring, 'green', False) else 0.0
        data[7] = 1.0 if getattr(tile.wiring, 'actuator', False) else 0.0
    
    return data


def optimized_array_to_tile_dict(data: np.ndarray) -> dict:
    """
    Convert 9-element array back to human-readable tile dictionary.
    
    Args:
        data: 9-element array
        
    Returns:
        Dictionary with tile properties
    """
    from .natural_ids import BLOCK_INDEX_TO_ID, WALL_INDEX_TO_ID
    
    # Block
    block_index = int(round(data[0]))
    block_index = np.clip(block_index, 0, 217)
    block_game_id = BLOCK_INDEX_TO_ID.get(block_index, 0)
    
    # Wall
    wall_index = int(round(data[2]))
    wall_index = np.clip(wall_index, 0, 76)
    wall_game_id = WALL_INDEX_TO_ID.get(wall_index, 0)
    
    # Liquid
    liquid_present = data[3] > 0.5
    liquid_type_idx = int(round(data[4]))
    liquid_type_names = ['NONE', 'WATER', 'LAVA', 'HONEY', 'SHIMMER']
    liquid_type = liquid_type_names[np.clip(liquid_type_idx, 0, 4)]
    
    return {
        'block_id': block_game_id,
        'block_shape': int(round(data[1] * 5)),
        'wall_id': wall_game_id,
        'liquid': liquid_type if liquid_present else 'NONE',
        'wire_red': bool(data[5] > 0.5),
        'wire_blue': bool(data[6] > 0.5),
        'wire_green': bool(data[7] > 0.5),
        'actuator': bool(data[8] > 0.5),
    }
