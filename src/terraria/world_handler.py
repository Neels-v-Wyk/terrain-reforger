"""
World loading and management utilities.

This module provides functions for loading and managing Terraria world files
using the lihzahrd library.
"""

from pathlib import Path
from typing import Union

import lihzahrd


def load_world(path: Union[str, Path]) -> lihzahrd.World:
    """
    Load a Terraria world file.
    
    Args:
        path: Path to the .wld world file
        
    Returns:
        Loaded world object from lihzahrd library
        
    Raises:
        FileNotFoundError: If the world file doesn't exist
        ValueError: If the file is not a valid Terraria world file
        
    Example:
        >>> world = load_world("worlds/MyWorld.wld")
        >>> print(f"World size: {world.size}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"World file not found: {path}")
    
    if path.suffix != '.wld':
        raise ValueError(f"Invalid world file extension: {path.suffix}. Expected '.wld'")
    
    try:
        return lihzahrd.World.create_from_file(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load world file: {e}") from e
