"""
Terraria world data representation module.

This module provides data structures for representing Terraria tiles and chunks
in a format suitable for machine learning applications.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np


@dataclass
class TileData:
    """
    Structured representation of a single Terraria tile with all its properties.
    
    Attributes:
        x: X-coordinate of the tile
        y: Y-coordinate of the tile
        block_type: Block type ID (0-700+)
        block_active: Whether the block exists
        block_shape: Shape of the block (0=normal, 1=half, 2-5=slopes)
        block_paint: Paint color (0-31, None if unpainted)
        block_illuminant: Whether the block glows
        block_echo: Whether the block has echo coating
        wall_type: Wall type ID (0-300+)
        wall_paint: Wall paint color (0-31, None if unpainted)
        wall_illuminant: Whether the wall glows
        wall_echo: Whether the wall has echo coating
        liquid_type: Liquid type (0=none, 1=water, 2=lava, 3=honey)
        liquid_amount: Liquid volume (0-255)
        wire_red: Red wire presence
        wire_blue: Blue wire presence
        wire_green: Green wire presence
        wire_yellow: Yellow wire presence
        actuator: Actuator presence
    """
    
    x: int
    y: int
    
    # Block properties
    block_type: Optional[int] = None
    block_active: bool = False
    block_shape: int = 0
    block_paint: Optional[int] = None
    block_illuminant: bool = False
    block_echo: bool = False
    
    # Wall properties
    wall_type: Optional[int] = None
    wall_paint: Optional[int] = None
    wall_illuminant: bool = False
    wall_echo: bool = False
    
    # Liquid properties
    liquid_type: int = 0
    liquid_amount: int = 0
    
    # Wiring properties
    wire_red: bool = False
    wire_blue: bool = False
    wire_green: bool = False
    wire_yellow: bool = False
    actuator: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tile data to dictionary format for inspection."""
        return {
            'position': (self.x, self.y),
            'block': {
                'type': self.block_type,
                'active': self.block_active,
                'shape': self.block_shape,
                'paint': self.block_paint,
                'illuminant': self.block_illuminant,
                'echo': self.block_echo
            },
            'wall': {
                'type': self.wall_type,
                'paint': self.wall_paint,
                'illuminant': self.wall_illuminant,
                'echo': self.wall_echo
            },
            'liquid': {
                'type': self.liquid_type,
                'amount': self.liquid_amount
            },
            'wiring': {
                'red': self.wire_red,
                'blue': self.wire_blue,
                'green': self.wire_green,
                'yellow': self.wire_yellow,
                'actuator': self.actuator
            }
        }


@dataclass
class ChunkData:
    """
    Structured representation of a chunk of Terraria world data.
    
    A chunk is a rectangular region of tiles with associated metadata like
    chests, signs, and tile entities.
    
    Attributes:
        x_start: Starting x-coordinate of the chunk
        y_start: Starting y-coordinate of the chunk
        width: Width of the chunk in tiles
        height: Height of the chunk in tiles
        tiles: 2D list of TileData objects (row-major order)
        chests: List of chests within this chunk
        signs: List of signs within this chunk
        tile_entities: List of tile entities within this chunk
    """
    
    x_start: int
    y_start: int
    width: int
    height: int
    tiles: List[List[TileData]] = field(default_factory=list)
    chests: List[Any] = field(default_factory=list)
    signs: List[Any] = field(default_factory=list)
    tile_entities: List[Any] = field(default_factory=list)
    
    def to_tensor(self) -> np.ndarray:
        """
        Convert chunk to multi-channel tensor representation for ML training.
        
        This representation captures all discrete states needed for terrain
        regeneration using diffusion models.
        
        Returns:
            Tensor of shape (height, width, 17) with the following channels:
            
            0: Block type (categorical, 0-700+)
            1: Block active (binary)
            2: Block shape (0-5: normal, half, 4 slopes)
            3: Block paint (0-31, 0=none)
            4: Block illuminant (binary)
            5: Block echo (binary)
            6: Wall type (categorical, 0-300+)
            7: Wall paint (0-31, 0=none)
            8: Wall illuminant (binary)
            9: Wall echo (binary)
            10: Liquid type (0=none, 1=water, 2=lava, 3=honey)
            11: Liquid amount (normalized 0-1)
            12-16: Wire colors (red, blue, green, yellow) + actuator (binary)
        """
        num_channels = 17
        tensor = np.zeros((self.height, self.width, num_channels), dtype=np.float32)
        
        for i, row in enumerate(self.tiles):
            for j, tile in enumerate(row):
                # Block channels (0-5)
                if tile.block_active and tile.block_type is not None:
                    tensor[i, j, 0] = tile.block_type
                tensor[i, j, 1] = 1.0 if tile.block_active else 0.0
                tensor[i, j, 2] = tile.block_shape
                tensor[i, j, 3] = tile.block_paint if tile.block_paint is not None else 0
                tensor[i, j, 4] = 1.0 if tile.block_illuminant else 0.0
                tensor[i, j, 5] = 1.0 if tile.block_echo else 0.0
                
                # Wall channels (6-9)
                tensor[i, j, 6] = tile.wall_type if tile.wall_type is not None else 0
                tensor[i, j, 7] = tile.wall_paint if tile.wall_paint is not None else 0
                tensor[i, j, 8] = 1.0 if tile.wall_illuminant else 0.0
                tensor[i, j, 9] = 1.0 if tile.wall_echo else 0.0
                
                # Liquid channels (10-11)
                tensor[i, j, 10] = tile.liquid_type
                tensor[i, j, 11] = tile.liquid_amount / 255.0 if tile.liquid_amount > 0 else 0.0
                
                # Wiring channels (12-16)
                tensor[i, j, 12] = 1.0 if tile.wire_red else 0.0
                tensor[i, j, 13] = 1.0 if tile.wire_blue else 0.0
                tensor[i, j, 14] = 1.0 if tile.wire_green else 0.0
                tensor[i, j, 15] = 1.0 if tile.wire_yellow else 0.0
                tensor[i, j, 16] = 1.0 if tile.actuator else 0.0
        
        return tensor
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this chunk."""
        tile_count = self.width * self.height
        active_blocks = sum(
            1 for row in self.tiles for tile in row if tile.block_active
        )
        walls = sum(
            1 for row in self.tiles for tile in row if tile.wall_type
        )
        liquids = sum(
            1 for row in self.tiles for tile in row if tile.liquid_type > 0
        )
        slopes = sum(
            1 for row in self.tiles for tile in row if tile.block_shape > 0
        )
        
        return {
            'position': (self.x_start, self.y_start),
            'size': (self.width, self.height),
            'total_tiles': tile_count,
            'active_blocks': active_blocks,
            'walls': walls,
            'liquids': liquids,
            'slopes': slopes,
            'chests': len(self.chests),
            'signs': len(self.signs),
            'tile_entities': len(self.tile_entities)
        }
